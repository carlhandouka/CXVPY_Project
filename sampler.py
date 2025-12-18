import numpy as np
from typing import Sequence, Optional, List

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False


# ------------------ Numba helpers (if available) ------------------

if NUMBA_AVAILABLE:
    @nb.njit(inline='always')
    def _build_allowed_candidates(bi: np.ndarray, kept_mask: np.ndarray, out_idx: np.ndarray) -> int:
        """
        bi, kept_mask: uint8 (0/1) of length universe.
        Writes allowed indices to out_idx and returns the count.
        """
        cnt = 0
        n = bi.shape[0]
        for v in range(n):
            if bi[v] == 1 and kept_mask[v] == 0:
                out_idx[cnt] = v
                cnt += 1
        return cnt

    @nb.njit(inline='always')
    def _partial_shuffle_pick_first_k(a: np.ndarray, n: int, k: int):
        """
        Partial Fisher–Yates: a[0:k] becomes a uniform k-subset of a[0:n].
        """
        for i in range(k):
            j = np.random.randint(i, n)  # [i, n)
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp

    @nb.njit
    def _chain_window_numba(bitsets_u8_window: np.ndarray, m: int,
                            kept_mask: np.ndarray, candidates: np.ndarray,
                            out_int32: np.ndarray) -> None:
        """
        bitsets_u8_window: (k, universe) uint8 0/1 for the contiguous window.
        out_int32: (k, m) where we write the result.
        """
        k = bitsets_u8_window.shape[0]
        universe = bitsets_u8_window.shape[1]

        # Step 0: sample m from the first list
        for v in range(universe):
            kept_mask[v] = 0
        cnt = _build_allowed_candidates(bitsets_u8_window[0], kept_mask, candidates)
        if cnt < m:
            raise ValueError("First list insufficient for m (no replacement).")
        _partial_shuffle_pick_first_k(candidates, cnt, m)
        for t in range(m):
            out_int32[0, t] = candidates[t]

        # Step 1..k-1
        for i in range(1, k):
            bi = bitsets_u8_window[i]
            kept_count = 0
            # keep survivors
            for t in range(m):
                val = out_int32[i-1, t]
                if bi[val] == 1:
                    out_int32[i, kept_count] = val
                    kept_count += 1

            need = m - kept_count
            if need > 0:
                # exclude already kept
                for v in range(universe):
                    kept_mask[v] = 0
                for t in range(kept_count):
                    kept_mask[out_int32[i, t]] = 1

                cnt = _build_allowed_candidates(bi, kept_mask, candidates)
                if cnt < need:
                    raise ValueError("Window list insufficient to reach m.")
                _partial_shuffle_pick_first_k(candidates, cnt, need)
                for t in range(need):
                    out_int32[i, kept_count + t] = candidates[t]

    @nb.njit
    def _numba_seed(seed: int) -> None:
        np.random.seed(seed)


# ------------------ NumPy Fallback (still very fast) ------------------

def _chain_window_numpy(bitsets_u8_window: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    bitsets_u8_window: (k, universe) uint8 0/1
    Returns (k, m) uint16.
    """
    k, universe = bitsets_u8_window.shape
    bitsets = bitsets_u8_window.astype(bool, copy=False)

    out = np.empty((k, m), dtype=np.uint16)

    # step 0
    b0 = bitsets[0]
    base_vals = np.flatnonzero(b0)
    if base_vals.size < m:
        raise ValueError("First list insufficient for m (no replacement).")
    sample = rng.choice(base_vals, size=m, replace=False).astype(np.uint16)
    out[0] = sample

    # step 1..k-1
    kept_mask = np.zeros(universe, dtype=bool)
    for i in range(1, k):
        bi = bitsets[i]
        kept = sample[bi[sample]]
        need = m - kept.size
        if need:
            kept_mask[:] = False
            kept_mask[kept] = True
            allowed = np.logical_and(bi, np.logical_not(kept_mask))
            candidates = np.flatnonzero(allowed)
            if candidates.size < need:
                raise ValueError("Window list insufficient to reach m.")
            repl = rng.choice(candidates, size=need, replace=False).astype(np.uint16)
            sample = np.concatenate([kept, repl])
        else:
            sample = kept
        out[i] = sample
    return out


# ------------------ Main Class ------------------

class WindowBitsetSampler:
    """
    Preprocesses your K lists (integer values >=0) into uint8 bitsets once
    and allows sampling on any contiguous window of length k:
      - step 0: sample m without replacement from the first list of the window
      - step i>0: keep survivors and replace missing ones from list_i avoiding duplicates
    Uses Numba JIT (compiled in __init__) if available, with NumPy fallback.
    """

    def __init__(self,
                 lists: Sequence[Sequence[int] | np.ndarray],
                 universe: Optional[int] = None,
                 use_numba: bool = True,
                 warmup: bool = True,
                 seed: Optional[int] = None):
        """
        lists: sequence of K lists/arrays with integer values >=0 (uniqueness not required).
        universe:
            - None -> automatically inferred as max_val + 1 from data
            - int  -> used as exclusive upper bound (values out of range are ignored)
        use_numba: try to use Numba (if installed).
        warmup: if True, compiles the Numba kernel in __init__ to avoid cost on first call.
        seed: for reproducibility (NumPy + Numba).
        """
        self.K = len(lists)

        # --- universe inference if not provided ---
        if universe is None:
            max_val = None
            for arr in lists:
                a = np.asarray(arr)
                if a.size:
                    # accept only non-negatives to define the range
                    a_nonneg = a[a >= 0]
                    if a_nonneg.size:
                        amax = int(a_nonneg.max())
                        max_val = amax if max_val is None else max(max_val, amax)
            if max_val is None:
                raise ValueError("Cannot infer 'universe': all lists are empty or negative. Pass universe explicitly.")
            universe = max_val + 1  # [0..max_val]
        self.universe = int(universe)

        # --- backend selection ---
        self._numba_enabled = bool(use_numba and NUMBA_AVAILABLE)

        # --- preprocess: K x universe uint8 (0/1), automatic dedup via assignment to 1 ---
        bitsets = np.zeros((self.K, self.universe), dtype=np.uint8)
        for i, arr in enumerate(lists):
            a = np.asarray(arr, dtype=np.int64).ravel()
            a = a[(a >= 0) & (a < self.universe)]
            if a.size:
                bitsets[i, np.unique(a)] = 1
        self._bitsets = bitsets

        # RNG and state
        self._rng = np.random.default_rng(seed)
        if self._numba_enabled:
            _numba_seed(int(seed if seed is not None else np.random.SeedSequence().entropy))
            # reusable workspace
            self._kept_mask_u8 = np.zeros(self.universe, dtype=np.uint8)
            self._candidates_i32 = np.empty(self.universe, dtype=np.int32)
            if warmup:
                # Compile for real universe dimensions
                dummy_out = np.empty((1, 1), dtype=np.int32)
                _chain_window_numba(self._bitsets[:1], 1,
                                    self._kept_mask_u8, self._candidates_i32, dummy_out)

    def seed(self, seed: int) -> None:
        """Resets the seed for NumPy and Numba."""
        self._rng = np.random.default_rng(seed)
        if self._numba_enabled:
            _numba_seed(int(seed))

    def sample_chain_window(self, start: int, k: int, m: int) -> np.ndarray:
        """
        Performs sampling on the contiguous window [start, start+k-1].
        Returns an array (k, m) uint16.
        """
        if not (0 <= start < self.K) or start + k > self.K:
            raise IndexError("Window out of bounds: start + k > K.")
        if m <= 0:
            raise ValueError("m must be > 0.")

        window = self._bitsets[start:start + k]  # (k, universe)

        if self._numba_enabled:
            out_i32 = np.empty((k, m), dtype=np.int32)
            _chain_window_numba(window, int(m),
                                self._kept_mask_u8, self._candidates_i32, out_i32)
            return out_i32.astype(np.uint16, copy=False)
        else:
            return _chain_window_numpy(window, int(m), self._rng)

    def sample_many_windows(self, starts: Sequence[int], k: int, m: int) -> List[np.ndarray]:
        """Convenient for calculating multiple windows in series; returns a list of (k, m)."""
        return [self.sample_chain_window(int(s), k, m) for s in starts]

    @property
    def K_lists(self) -> int:
        return self.K

    @property
    def bitsets_view(self) -> np.ndarray:
        """Read-only access to preprocessed bitsets (K, universe)."""
        return self._bitsets


class CommonSampler:
    """
    Identifies the intersection of available elements (e.g. stocks) across a contiguous window
    and samples a fixed subset from that intersection.
    Useful for ensuring the selected assets are available throughout the entire batch window.
    """
    def __init__(self, lists: Sequence[Sequence[int] | np.ndarray], rng=None):
        self.sets = list(map(set,lists))
        self._rng = rng if rng is not None else np.random.default_rng()

    def __common_observed_indices(self, start,end) -> np.ndarray:
        sets = self.sets[start:end]
        common = sets[0].intersection(*sets[1:])
        return list(common)
    
    def sampler(self, start:int, k:int, m:int) -> np.ndarray:
        """
        Samples 'm' elements common to all lists in [start, start + k].
        
        Args:
            start: Start index of the window.
            k: Length of the window (batch size).
            m: Number of elements to select.
            
        Returns:
            (k, m) array with the same selected elements repeated.
        """
        common = self.__common_observed_indices(start,start+k)
        
        if len(common) < m:
            raise ValueError("Not enough common elements to sample from.")
        selected = self._rng.choice(common, size=m, replace=False)
        return np.tile(selected, (k,1))
