import os
import numpy as np
import pandas as pd
from glob import glob

from joblib import Parallel, delayed

from config.config_loader import load_config
from eye_feature_engineering.eye_feature_utils.calc_utils import calculate_time_delta, \
    calculate_velocity_and_acceleration
from eye_feature_engineering.eye_feature_utils.feature_utils import get_features, get_event_basic_features
from utils.general_utils import read_eye_gaze_into_df, read_remodav_event_into_df

# ------------------------------------------------------------------------------
# setting location
# setting parallelization
CONFIG_DATA = load_config(os.path.join("..", "config", "feature_engineering_config.json"))

NUM_CORES = CONFIG_DATA["num_cores"]

PATH_IN = CONFIG_DATA["path_in"]
GAZE_FILE_PATTERN = CONFIG_DATA["gaze_file_pattern_to_read"]
EVENT_FILE_PATTERN = CONFIG_DATA["event_file_pattern_to_read"]
PATH_OUT = CONFIG_DATA["path_out"]
SCENARIOS = CONFIG_DATA["scenarios"]
STATES = CONFIG_DATA["states"]
IDS = CONFIG_DATA["ids"]

EYE_TRACKER_HZ = CONFIG_DATA["eye_tracker_hz"]

# ------------------------------------------------------------------------------


print('loaded dependencies')
# =============================================================================
# settings

WINDOW_SIZE_S = CONFIG_DATA["sliding_window_settings"]["window_size_s"]


# ==============================================================================
# functions

def save_one_participant_csv(gaze_filepath, event_filepath):
    print("Start processing file", gaze_filepath)
    scenario = [s for s in SCENARIOS if s in gaze_filepath][0]
    state = [s for s in STATES if s in gaze_filepath][0]
    d_id = [i for i in IDS if i in gaze_filepath][0]
    print("Scenario:", scenario)
    print("State:", state)
    print("ID:", d_id)

    print('Loading and recalculating timestamps gaze...')
    print(gaze_filepath)
    # read eye-tracking gaze data into dataframe
    gaze_df = read_eye_gaze_into_df(gaze_filepath, "feature")

    # MM interpolate
    target_index = pd.date_range(start=gaze_df.index[0].ceil('s'),  # to account for a/v
                                 end=gaze_df.index[-1].floor('s'),
                                 freq='%dus' % (1000000 / EYE_TRACKER_HZ))  # microsecs
    gaze_df = gaze_df.reindex(index=gaze_df.index.union(target_index).drop_duplicates())
    gaze_df.interpolate(method='time', limit=int(EYE_TRACKER_HZ), inplace=True)
    gaze_df = gaze_df.reindex(target_index)

    gaze_df.dropna(inplace=True)
    # MM end

    print('Loading event file...')
    print(event_filepath)
    event_df = read_remodav_event_into_df(event_filepath, ["FIXA", "SACC", "PURS", "ISAC", "BLINK"])

    # ------- ------------------------------------------------------------------
    print('Calculating timedelta...')
    gaze_df = calculate_time_delta(gaze_df)

    # ------- ------------------------------------------------------------------
    print('Calculating velocity and acceleration...')
    gaze_df = calculate_velocity_and_acceleration(gaze_df, ["x", "y"])
    gaze_df = calculate_velocity_and_acceleration(gaze_df, ["x", "y", "z"], "head")

    # neglect NaNs in first 2 samples of recording
    gaze_df = gaze_df[2:]

    # --------------------------------------------------------------------------
    print(' - raw data length: ', len(gaze_df))
    # --------------------------------------------------------------------------
    print('Filtering dt...')
    threshold_slow = 1 / (EYE_TRACKER_HZ - 30)  # (60Hz = 0.0166)
    threshold_fast = 1 / (EYE_TRACKER_HZ + 30)  # (120Hz = 0.0083)

    idx_too_slow_or_fast = ((gaze_df['dt'] > threshold_slow) | (gaze_df['dt'] < threshold_fast))
    idx_shifted_for_vel = idx_too_slow_or_fast.shift(periods=1)
    idx_shifted_for_acc = idx_too_slow_or_fast.shift(periods=2)

    gaze_df = gaze_df[~(idx_too_slow_or_fast | idx_shifted_for_vel | idx_shifted_for_acc)]
    print(' - filtered data length: ', len(gaze_df))

    # delete delta time
    gaze_df = gaze_df.drop(
        columns=['dt'])  # --> can also do it before model input but then for all statistical features

    # --------------------------------------------------------------------------
    print('Aggregating features...')

    # calculate L2 norm as combination of features
    # gaze_df['l2'] = np.linalg.norm(gaze_df.to_numpy(), axis=1)

    # calculate statistical features
    gaze_agg = get_features(gaze_df[['x', 'y', 'v_x', 'v_y', 'v_x_y', 'a_x', 'a_y', 'a_x_y']],
                            window_size_s=WINDOW_SIZE_S,
                            num_cores=NUM_CORES)
    gaze_agg = gaze_agg.add_prefix('gaze+')

    head_agg = get_features(gaze_df[
                                ['head_x', 'head_y', 'head_z',
                                 'v_head_x', 'v_head_y', 'v_head_z',
                                 'v_head_x_y', 'v_head_x_z', 'v_head_y_z', 'v_head_x_y_z',
                                 'a_head_x', 'a_head_y', 'a_head_z',
                                 'a_head_x_y', 'a_head_x_z', 'a_head_y_z', 'a_head_x_y_z']],
                            window_size_s=WINDOW_SIZE_S,
                            num_cores=NUM_CORES)
    head_agg = head_agg.add_prefix('head+')

    start_time_gaze, end_time_gaze = min(gaze_df.index), max(gaze_df.index)

    fixation_agg_bin = get_event_basic_features(event_df, "FIXA",
                                                ["duration", "amp", "peak_vel", "med_vel", "avg_vel"],
                                                start_time_gaze, end_time_gaze,
                                                window_size_s=WINDOW_SIZE_S,
                                                num_cores=NUM_CORES)
    fixation_agg_bin = fixation_agg_bin.add_prefix('fixation+')

    saccades_agg_bin = get_event_basic_features(event_df, "SACC",
                                                ["duration", "amp", "peak_vel", "med_vel", "avg_vel"],
                                                start_time_gaze, end_time_gaze,
                                                window_size_s=WINDOW_SIZE_S,
                                                num_cores=NUM_CORES)
    saccades_agg_bin = saccades_agg_bin.add_prefix("saccades+")

    all_agg = pd.concat(
        [gaze_agg, head_agg, fixation_agg_bin, saccades_agg_bin],
        axis=1,
        sort=False)
    all_agg = all_agg.loc[gaze_agg.index]

    if "num_samples" in all_agg:
        all_agg["ratio_samples_available"] = all_agg["num_samples"] / (EYE_TRACKER_HZ * 60 * WINDOW_SIZE_S)

    all_agg.index = all_agg.index.floor('s')  # 'nicify' indices

    all_agg["state"] = state
    all_agg["scenario"] = scenario

    all_agg["id"] = d_id

    write_filepath = os.path.join(PATH_OUT, f"{d_id}_{scenario}_{WINDOW_SIZE_S}_{state}_single.csv")
    all_agg.to_csv(write_filepath)

    return


def save_all_participant(file_name_difference="aggregated"):
    print('----------------------------------------')
    print('----------------------------------------')
    save_path = os.path.join(PATH_OUT,
                             file_name_difference + '_all_participants_' + str(WINDOW_SIZE_S) + 's.csv')
    id_name = '*_single.csv'

    if os.path.isfile(save_path):
        os.remove(save_path)
    print('Summarizing all participants')

    participant_paths = glob(os.path.join(PATH_OUT, id_name))
    participant_paths = [p for p in participant_paths if any(p_id in p for p_id in IDS)]
    participant_paths = np.sort(participant_paths)
    print('Number of participants: ' + str(participant_paths.size))

    # initialize dataframe
    data_df = pd.DataFrame()
    for participant_path in participant_paths:
        print('----------------------------------------')
        print('adding participant: ', participant_path)
        participant_df = pd.read_csv(participant_path, parse_dates=[0], index_col=0)
        print('participant_df: ', participant_df.shape)
        data_df = pd.concat([data_df, participant_df], axis=0)
        print('data_df: ', data_df.shape)

    # save as csv per participant
    data_df.to_csv(save_path)
    print('Saved in: ', save_path)

    for participant_path in participant_paths:
        os.remove(participant_path)

    return


# ==============================================================================
# MAIN CODE
if __name__ == '__main__':

    gaze_filepaths = glob(os.path.join(PATH_IN, GAZE_FILE_PATTERN))
    if len(gaze_filepaths) == 0:
        print(PATH_IN, f"has no files with the pattern {GAZE_FILE_PATTERN}")

    event_filepaths = glob(os.path.join(PATH_IN, EVENT_FILE_PATTERN))
    if len(event_filepaths) == 0:
        print(PATH_IN, f"has no files with the pattern {EVENT_FILE_PATTERN}")

    if len(gaze_filepaths) != len(event_filepaths):
        print("Missing files. Gaze and event files have an unequal length.")

    print(f"Processing {len(gaze_filepaths)} files from {PATH_IN}...")

    Parallel(n_jobs=NUM_CORES, backend='multiprocessing')(
        delayed(save_one_participant_csv)(g, e) for g, e in zip(gaze_filepaths, event_filepaths))

    save_all_participant(file_name_difference="aggregated")

    print(f"{len(gaze_filepaths)} files processed. Output stored in {PATH_OUT}.")
