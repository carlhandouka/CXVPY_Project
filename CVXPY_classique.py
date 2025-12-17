import cvxpy as cp
import numpy as np


def generate_spd_matrix(n: int, seed: int | None = None, epsilon: float = 1e-3) -> np.ndarray:
    """
    Crée une matrice symétrique définie positive de taille n×n.

    Parameters
    ----------
    n : int
        Dimension de la matrice.
    seed : int | None
        Graine pour la reproductibilité. Laisser None pour une génération aléatoire.
    epsilon : float
        Joue le rôle de régularisation sur la diagonale pour garantir la positivité définie.

    Returns
    -------
    np.ndarray
        Matrice symétrique définie positive.
    """
    if n <= 0:
        raise ValueError("n doit être strictement positif.")

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T
    sigma += epsilon * np.eye(n)
    return sigma


def minimise_variance(sigma: np.ndarray, allow_short_selling: bool = False) -> np.ndarray:
    """
    Résout un problème de portefeuille à variance minimale.
    minimise   wᵀ Σ w
    s.c.       1ᵀ w = 1
               w ≥ 0           (sauf si allow_short_selling=True)
    Parameters
    ----------
    sigma : np.ndarray
        Matrice de covariance (symétrique définie positive) de dimension n×n.
    allow_short_selling : bool
        Si False, impose w >= 0. Sinon, autorise les positions vendeuses.
    Returns
    -------
    np.ndarray
        Les poids optimaux du portefeuille minimisant la variance.
    """
    sigma = np.asarray(sigma)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma doit être une matrice carrée.")
    n_assets = sigma.shape[0]
    w = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = [cp.sum(w) == 1]
    if not allow_short_selling:
        constraints.append(w >= 0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimisation échouée (status={problem.status}).")
    return w.value


if __name__ == "__main__":
    sigma_example = generate_spd_matrix(n=3, seed=1)
    weights = minimise_variance(sigma_example,True)
    print("Poids optimaux (variance minimale, sans ventes à découvert) :", weights)
