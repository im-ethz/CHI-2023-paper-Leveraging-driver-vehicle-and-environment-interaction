import multiprocessing

from joblib import Parallel, delayed
import numpy as np
from scipy.stats import skew, kurtosis, iqr
from datetime import timedelta

import pandas as pd


def get_features(data: pd.DataFrame, window_size_s: int = 60, num_cores: int = 0, step_size: str = "1S"):
    """
    Function to get aggregated features in parallel implementation
    data: (pd.DataFrame) data of whole dataframe to aggregate (no columns 'id', 'label', 'scenario')
    epoch_width: (int) time window in [s] to aggregate
    num_cores: (int) number of CPU cores to be used
    :return: (pd.DataFrame) data of aggregated features
    """
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()
    print('using # cores: ', num_cores)

    input_data = data.copy()
    # input_data['l2'] = np.linalg.norm(input_data.to_numpy(), axis=1)

    start_time = input_data.index.min()
    end_time = input_data.index.max()

    inputs = pd.date_range(start_time, end_time, freq=step_size)

    results = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(get_sliding_window)(input_data, window_size_s=window_size_s, i=k) for k in inputs)
    results = pd.DataFrame(list(filter(None, results)))  # filter out None values
    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def get_event_basic_features(event_data: pd.DataFrame, event_type: str, columns: [str], start_time: pd.Timestamp,
                             end_time: pd.Timestamp, window_size_s: int = 60, num_cores: int = 0,
                             step_size: str = "1S"):
    """
    Function to get aggregated features in parallel implementation
    event_data: (pd.DataFrame) data of whole event dataframe to aggregate
    event_type: (str) event to calculate features for
    columns: ([str]) columns to calculate features on
    start_time: (pd.Timestamp) start time of the gaze data, needed for creating correct sliding windows
    end_time: (pd.Timestamp) end time of the gaze data, needed for creating correct sliding windows
    epoch_width: (int) time window in [s] to aggregate
    num_cores: (int) number of CPU cores to be used
    use_region_data: (bool) whether the features should be calculated per region
    :return: (pd.DataFrame) data of aggregated features incl. a 'count' column
    """
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()
    print('using # cores: ', num_cores)

    event_input_data = event_data[event_data["label"] == event_type][columns].copy()

    inputs = pd.date_range(start_time, end_time, freq=step_size)

    results = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(get_sliding_window)(event_input_data, window_size_s=window_size_s, i=k) for
        k in inputs)
    results = pd.DataFrame(list(filter(None, results)))  # filter out None values

    results.set_index('datetime', inplace=True)
    results.sort_index(inplace=True)

    return results


def get_sliding_window(data: pd.Series, window_size_s: int, i: int):
    """
    Function to get aggregated features in parallel implementation
    data: (pd.DataFrame) data of whole dataframe to aggregate (no columns 'id', 'label', 'scenario')
    epoch_width: (int) time window in [s] to aggregate
    i: (int) index of start frame of aggregation
    :return: (pd.DataFrame) data of aggregated features with column 'num_samples'
    """
    min_timestamp = i
    max_timestamp = min_timestamp + timedelta(seconds=window_size_s)

    results = {
        'datetime': min_timestamp,
    }

    relevant_data = data.loc[(data.index >= min_timestamp) & (data.index < max_timestamp)]

    for column in relevant_data.columns:
        column_results = get_stats(relevant_data[column], column)
        results.update(column_results)

    return results


def get_stats(data, key_suffix: str = None):
    """
    Function defining the statistical measures considered for aggregation
    :return: (pd.DataFrame) data of aggregated featues with column 'num_samples'
    """
    results = {
        'mean': np.nan,
        'std': np.nan,
        'q5': np.nan,
        'q95': np.nan,
        'power': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan,
    }

    if len(data) > 0:
        results['mean'] = np.mean(data)
        results['std'] = np.std(data)
        results['q5'] = np.quantile(data, 0.05)
        results['q95'] = np.quantile(data, 0.95)
        results['skewness'] = skew(data)
        results['kurtosis'] = kurtosis(data)

    if key_suffix is not None:
        results = {k + '_' + key_suffix: v for k, v in results.items()}

    return results
