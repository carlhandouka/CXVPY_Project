from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pipeline.sampler import CommonSampler, WindowBitsetSampler
DATA = f'DATA/vanilla_returns_top_1000_without_NaN_dtin_max_1200.joblib'


def _max_lookback(n_days_in: int | None = None, n_days_in_range: Tuple[int, int] | None = None) -> int:
    """Return the maximum lookback needed from fixed n_days_in or the upper bound of a range."""
    if n_days_in is None and n_days_in_range is None:
        raise ValueError(
            "Provide at least one among n_days_in or n_days_in_range.")
    return int(n_days_in if n_days_in is not None else n_days_in_range[1])


def prepare_dataset(date_bounds: Tuple[str, str], market_cap_range: Tuple[int, int] = (0, 3000), n_days_out: int = 5, shift: int = 1, filename=DATA,
                    n_days_in: int | None = None, n_days_in_range: Tuple[int, int] | None = None):
    """
    Prepares the dataset for the given parameters.
    Parameters:
    n_days_in (int, optional): Input window.
    n_days_in_range (tuple, optional): Range of possible input windows.
    date_bounds (tuple): A tuple containing the start and end dates (d0, d1) for the data selection.
    market_cap_range (tuple, optional): Slice of stocks to select by Market Cap. Defaults to (0, 3000).
    filename (str, optional): Path to the joblib file containing the dataset. Defaults to DATA.
    Returns:
    tuple: A tuple containing:
        - returns (numpy.ndarray): The array of returns for the selected date range.
        - date_stock_mapping (list): A list of available stocks for each date in the selected date range.
    """
    lookback = _max_lookback(
        n_days_in=n_days_in, n_days_in_range=n_days_in_range)

    start_date, end_date = date_bounds

    bundle = joblib.load(filename, mmap_mode='r')

    returns = bundle.returns
    returns.index = pd.to_datetime(returns.index)
    returns_cols_numeric = pd.api.types.is_numeric_dtype(returns.columns)
    if hasattr(bundle, 'returns') and hasattr(bundle.returns, '_mmap'):
        bundle.returns._mmap.close()

    available_stocks = bundle.available_stocks
    if hasattr(bundle, 'available_stocks') and hasattr(bundle.available_stocks, '_mmap'):
        bundle.available_stocks._mmap.close()

    available_stocks = available_stocks.iloc[:,
                                             market_cap_range[0]:market_cap_range[1]]
    available_stocks.index = pd.to_datetime(available_stocks.index)

    if returns_cols_numeric:
        available_stocks = available_stocks.apply(pd.to_numeric)
    else:
        returns.columns = returns.columns.astype(str)
        available_stocks = available_stocks.astype(str)

    codes, unique_stocks = pd.factorize(available_stocks.to_numpy().ravel())
    unique_stocks = pd.to_numeric(
        unique_stocks) if returns_cols_numeric else unique_stocks.astype(str)

    # Rimappiamo i codici alla forma originale del DataFrame
    available_stocks = pd.DataFrame(codes.reshape(available_stocks.shape),
                                    index=available_stocks.index,
                                    columns=available_stocks.columns)

    # Seleziona le colonne di returns in base ai nomi unici ottenuti
    returns = returns.loc[:, unique_stocks]
    available_stocks = available_stocks.loc[start_date:end_date]
    # rimuovi le ultime osservazioni non utilizzabili in-sample
    available_stocks = available_stocks.iloc[:-n_days_out-shift]

    if returns.loc[start_date:available_stocks.index[0]].shape[0] > 1:
        raise ValueError("The start date is earlier than the first available date in the available_stocks data: {}".format(
            available_stocks.index[0]))

    if available_stocks.shape[0] == 0:
        raise ValueError("No data available for the specified date range.")

    first_valid_index = returns.index.get_indexer(
        [start_date], method="bfill")[0]
    if first_valid_index < 0:
        raise ValueError("No valid index found for the specified start date.")

    if first_valid_index - lookback < 0:
        raise ValueError(
            "Requested lookback ({}) exceeds available data start.".format(lookback))
    start_cal = returns.index[first_valid_index - lookback]
    returns = np.ascontiguousarray(returns.loc[start_cal:end_date].to_numpy())
    available_stocks = np.ascontiguousarray(available_stocks.to_numpy())

    return returns, available_stocks, unique_stocks


def real_data_producer(historicalData: np.ndarray, available_stocks: np.ndarray, batch_size: int,  n_days_out: int, n_days_in: int = None, n_days_in_range: Tuple[int, int] = None,
                       n_stocks: int = None, n_stocks_range: Tuple[int, int] = None,  shift: int = 1,
                       sequential: bool = False, return_generator: bool = False,  common_stocks: bool = False,
                       rng: np.random.Generator = None, dtype: tf.DType = tf.float32):
    """
    Generates a TensorFlow dataset that produces batches of real stock data for training models.

    Args:
        historicalData (np.ndarray): A 2D array of historical stock data vanilla returns with shape (time, stocks).
        available_stocks (list): A list where each element is a list of available stock indices at a given time.
        batch_size (int): The number of samples per batch.
        n_days_out (int): The number of days of future data to use as output.
        n_days_in (int, optional): The number of days of historical data to use as input.
        n_days_in_range (Tuple[int, int], optional): Range (min, max) for the number of days to include in the input data. If None, all n_days_in are used. If range is provided, each batch will select a random number of days within that range.
        n_stocks (int, optional): The number of stocks to select. If None, n_stocks_range must be provided.
        n_stocks_range (Tuple[int, int], optional): Range (min, max) for the number of stocks to select. If None, n_stocks must be provided. If a range is provided, each batch will select a random number of stocks within that range.
        shift (int, optional): The number of days to shift the output data. This will cancel the lookahead bias. Defaults to 1.
        sequential (bool, optional): If True, the selected timesteps in a batch will be sequential. It will try to preserve the stocks within the batch without selection bias. If False, they will be randomly selected.
        common_stocks (bool, optional): If True, uses a common stock sampler across timesteps. Warning: This will imply a selection bias.
        return_generator (bool, optional): If True, returns the generator function instead of a tf.data.Dataset.
        rng (np.random.Generator, optional): A random number generator instance. If None, a default generator is used.
        dtype (tf.DType, optional): Data type for the output tensors.

    Raises:
        ValueError: If both n_stocks and n_stocks_range are None.
        ValueError: If sequential mode is used with n_days_out > 1.

    Returns:
        tf.data.Dataset or generator: A TensorFlow dataset (or generator if return_generator=True) that yields tuples of (returns_in, returns_out), where:
            - returns_in (tf.Tensor): A tensor of shape (batch_size, n_stocks, n_days_in) containing the input returns.
            - returns_out (tf.Tensor): A tensor of shape (batch_size, n_stocks, n_days_out) containing the output returns.
    """

    if n_stocks is None and n_stocks_range is None:
        raise ValueError("Both n_stocks and n_stocks_range cannot be None")
    if n_stocks is not None and n_stocks_range is not None:
        raise ValueError(
            "Only one of n_stocks or n_stocks_range should be provided")

    if n_days_in is None and n_days_in_range is None:
        raise ValueError("Both n_days_in and n_days_in_range cannot be None")
    if n_days_in is not None and n_days_in_range is not None:
        raise ValueError(
            "Only one of n_days_in or n_days_in_range should be provided")

    if sequential and n_days_out > 1:
        raise ValueError(
            "Sequential mode is not supported for n_days_out > 1. Please set sequential=False.")

    if rng is None:
        rng = np.random.default_rng()

    available_stocks = np.ascontiguousarray(available_stocks)
    lookback = _max_lookback(
        n_days_in=n_days_in, n_days_in_range=n_days_in_range)
    base_offset = lookback
    if base_offset <= 0:
        raise ValueError("Computed base_offset <= 0, check dataset alignment.")
    output_offsets = np.arange(
        shift, n_days_out + shift, dtype=np.int64).reshape(1, 1, -1)

    max_timestep = historicalData.shape[0] - \
        (base_offset + n_days_out + shift) + 1
    if max_timestep <= 0:
        raise ValueError(
            "Not enough timesteps for the requested n_days_out and shift.")
    usable_timesteps = min(len(available_stocks), max_timestep)
    if usable_timesteps <= 0:
        raise ValueError(
            "No usable timesteps available after aligning historical data with requested horizons.")
    available_stocks = available_stocks[:usable_timesteps]
    n_timesteps, _ = available_stocks.shape

    # Stock sampler for sequential selection
    if sequential:
        if batch_size > n_timesteps:
            raise ValueError(
                "Batch size larger than the number of usable timesteps.")
        if common_stocks:
            sampler = CommonSampler(available_stocks, rng=rng).sampler
        else:
            sampler = WindowBitsetSampler(
                available_stocks, use_numba=True, warmup=True, seed=rng.integers(0, 2**31)).sample_chain_window

    def data_generator():

        effective_n_days_in = lookback
        base_input_offsets = np.arange(-effective_n_days_in + 1,
                                       1, dtype=np.int64).reshape(1, 1, -1)

        while True:

            # Determine number of stocks for this batch
            if n_stocks_range is not None:
                n_stocks_local = rng.integers(*n_stocks_range, size=1)[0]
            else:
                n_stocks_local = n_stocks

            # Select timesteps and stocks
            if sequential:
                # Sequential selection
                selected_timesteps = np.arange(
                    batch_size) + rng.integers(0, n_timesteps - batch_size + 1)
                selected_stocks = sampler(
                    start=selected_timesteps[0], k=batch_size, m=n_stocks_local)
            else:
                # Random selection (per-row choice, fast enough and avoids large perms)
                selected_timesteps = rng.integers(
                    0, n_timesteps, size=batch_size)
                selected_stocks = np.empty(
                    (batch_size, n_stocks_local), dtype=available_stocks.dtype)
                for i, t in enumerate(selected_timesteps):
                    selected_stocks[i] = rng.choice(
                        available_stocks[t], n_stocks_local, replace=False)

            # Adjust time offsets if n_days_range is specified
            if n_days_in_range is not None:
                n_days_in_local = rng.integers(
                    n_days_in_range[0], n_days_in_range[1] + 1)
                input_offsets = np.arange(-n_days_in_local + 1,
                                          1, dtype=np.int64).reshape(1, 1, -1)
            else:
                n_days_in_local = effective_n_days_in
                input_offsets = base_input_offsets

            anchor_indices = selected_timesteps + base_offset
            time_indices_in = anchor_indices[:, None, None] + input_offsets
            time_indices_out = anchor_indices[:, None,
                                              None] + output_offsets.reshape(1, 1, -1)

            returns_in = historicalData[time_indices_in,
                                        selected_stocks[..., None]]
            returns_out = historicalData[time_indices_out,
                                         selected_stocks[..., None]]

            yield returns_in, returns_out

    if return_generator:
        return data_generator

    n = n_stocks if n_stocks else None
    t = None if n_days_in_range is not None else n_days_in if n_days_in else lookback

    # Create a dataset from the generator function
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=[batch_size, n, t], dtype=dtype),
            tf.TensorSpec(shape=[batch_size, n, n_days_out], dtype=dtype)
        )
    )
    # Prefetch the dataset to improve performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def real_data_pipeline(batch_size: int, date_bounds: Tuple[str, str], n_days_out: int, n_days_in: int = None, n_days_in_range: Tuple[int, int] = None, shift: int = 1,
                       n_stocks: int = None, n_stocks_range: Tuple[int, int] = None, market_cap_range: Tuple[int, int] = (0, 1000),
                       sequential: bool = False, return_generator: bool = False,  common_stocks: bool = False,
                       rng: np.random.Generator = None, dtype: tf.DType = tf.float32, filename: str = DATA):
    """
    High-level wrapper that prepares data and returns a producer (or tf.data.Dataset) for real stock returns.

    Args:
        batch_size (int): Number of samples per batch.
        date_bounds (Tuple[str, str]): Tuple containing start and end dates (d0, d1) of the out-of-sample window.
        n_days_out (int): Number of days of future data to use as output.
        n_days_in (int, optional): Number of days of historical data to use as input.
        n_days_in_range (Tuple[int, int], optional): Range (min, max) for number of days to include in input data.
        shift (int, optional): Number of days to shift the output data. Defaults to 1. This will cancel the lookahead bias.
        n_stocks (int, optional): Number of stocks to select.
        n_stocks_range (Tuple[int, int], optional): Range (min, max) for number of stocks to select.
        market_cap_range (Tuple[int, int], optional): Slice of stocks to select by Market Cap. Defaults to (0, 1000).
        sequential (bool, optional): If True, selected timesteps in a batch will be sequential. It will try to preserve the stocks within the batch without selection bias. 
        common_stocks (bool, optional): If True, uses a common stock sampler across timesteps. Warning: This will imply a selection bias.
        return_generator (bool, optional): If True, returns the generator function instead of a tf.data.Dataset.
        rng (np.random.Generator, optional): Random number generator instance. If None, a default generator is used.
        dtype (tf.DType, optional): Data type for the output tensors.
        filename (str, optional): Path to the joblib file containing the dataset. Defaults to DATA.
    Returns:
        tf.data.Dataset or generator: A TensorFlow dataset (or generator if return_generator=True) that yields tuples of (returns_in, returns_out).

    Raises:
        ValueError: If both n_stocks and n_stocks_range are None.
        ValueError: If both n_days_in and n_days_in_range are None.

    Example (generator):
        >>> producer = real_data_pipeline(
        ...     batch_size=32,
        ...     date_bounds=('1995-01-01','2015-01-01'),
        ...     n_days_out=5,
        ...     n_days_in=800,
        ...     n_stocks=50,
        ...     market_cap_range=(0,1000),
        ...     shift=1,
        ...     return_generator=True,
        ...     rng=np.random.default_rng(0)
        ... )
        >>> (rin, nan_mask), rout = next(producer())

    Example (tf.data.Dataset):
        >>> dataset = real_data_pipeline(
        ...     batch_size=32,
        ...     date_bounds=('1995-01-01','2015-01-01'),
        ...     n_days_out=5,
        ...     n_days_in_range=(600, 800),
        ...     n_stocks=50,
        ...     shift=1,
        ...     return_generator=False
        ... )
        >>> (rin, nan_mask), rout = next(iter(dataset))
    """

    historicalData, available_stocks, _ = prepare_dataset(date_bounds=date_bounds, market_cap_range=market_cap_range, n_days_out=n_days_out,
                                                          shift=shift, n_days_in=n_days_in, n_days_in_range=n_days_in_range, filename=filename)

    producer = real_data_producer(historicalData=historicalData, available_stocks=available_stocks,
                                  batch_size=batch_size,  n_days_out=n_days_out, n_days_in=n_days_in, n_days_in_range=n_days_in_range,
                                  n_stocks=n_stocks, n_stocks_range=n_stocks_range,  shift=shift,
                                  sequential=sequential, return_generator=return_generator,  common_stocks=common_stocks,
                                  rng=rng, dtype=dtype)
    return producer
