import sys
import os

from config.config_loader import load_config

import remodnav as rem
import pandas as pd

from utils.general_utils import read_eye_gaze_into_df, calc_sort_time

from glob import glob
import numpy as np

import math

from joblib import Parallel, delayed

CONFIG_DATA = load_config(os.path.join("..", "config", "remodnav_config.json"))

NUM_CORES = CONFIG_DATA["num_cores"]

DISTANCE_TO_SCREEN_CM_DICT = CONFIG_DATA["distance_to_screen"]

PATH_IN = CONFIG_DATA["path_in"]
GAZE_FILE_PATTERN = CONFIG_DATA["gaze_file_pattern_to_read"]
PATH_OUT = CONFIG_DATA["path_out"]

EYE_TRACKER_HZ = CONFIG_DATA["eye_tracker_hz"]

VELOCITY_THRESHOLD_START_VELOCITY = CONFIG_DATA["velocity_threshold_start_velocity"]
PURSUIT_VELOCITY_THRESHOLD = CONFIG_DATA["pursuit_velocity_threshold"]
SAVGOL_TARGET_SAMPLES = CONFIG_DATA["savgol_target_samples"]
MEDIAN_FILTER_TARGET_LENGTH_IN_S = CONFIG_DATA["median_filter_target_length_in_s"]
SACCADE_CONTEXT_WINDOW_LENGTH = CONFIG_DATA["saccade_context_window_length"]

MAX_DURATION_BLINK_S = CONFIG_DATA["max_duration_blink_s"]
MIN_DURATION_NAN_EVENT = CONFIG_DATA["min_duration_nan_event"]

SCREEN_CM_HEIGHT = CONFIG_DATA["simulator_screen_size"]["cm_height"]
SCREEN_PX_HEIGHT = CONFIG_DATA["simulator_screen_size"]["px_height"]
SCREEN_PX_WIDTH = CONFIG_DATA["simulator_screen_size"]["px_width"]


def apply_remodnav(tmp_read_filepath, tmp_store_filepath, distance_to_screen_cm=80,
                   savgol_target_samples=None,
                   median_filter_length_in_s=None,
                   velocity_threshold_start_velocity=None,
                   pursuit_velocity_threshold=None,
                   saccade_context_window_length=None,
                   remove_tmp_store_file=True):
    if not os.path.isfile(tmp_read_filepath):
        raise OSError("File", tmp_read_filepath, "is missing")

    px2deg = math.degrees(math.atan2(.5 * SCREEN_CM_HEIGHT, distance_to_screen_cm)) / (
            .5 * SCREEN_PX_WIDTH)

    variables_remodnav = [tmp_read_filepath, tmp_store_filepath, str(px2deg), str(EYE_TRACKER_HZ)]

    if savgol_target_samples is not None:
        savgol_length = round(savgol_target_samples / EYE_TRACKER_HZ, 2)
    else:
        savgol_length = round(3 / EYE_TRACKER_HZ, 2)
    variables_remodnav.extend(["--savgol-length", str(savgol_length)])

    if median_filter_length_in_s is not None:
        median_filter_length = math.ceil((median_filter_length_in_s / EYE_TRACKER_HZ) * 1000) / 1000
        variables_remodnav.extend(['--median-filter-length', str(median_filter_length)])
    if velocity_threshold_start_velocity is not None:
        variables_remodnav.extend(['--velthresh-startvelocity', str(velocity_threshold_start_velocity)])
    if pursuit_velocity_threshold is not None:
        variables_remodnav.extend(['--pursuit-velthresh', str(pursuit_velocity_threshold)])
    if saccade_context_window_length is not None:
        variables_remodnav.extend(['--saccade-context-window-length', str(saccade_context_window_length)])
    if EYE_TRACKER_HZ == 60:
        variables_remodnav.extend(['--dilate-nan', str(0.02)])
        variables_remodnav.extend(['--min-saccade-duration', str(0.02)])

    print("Use Remodnav with the following args:", variables_remodnav)
    rem.main([sys.argv[0]] + variables_remodnav)

    df_remodnav = pd.read_csv(tmp_store_filepath, sep="\t")
    if remove_tmp_store_file:
        os.remove(tmp_store_filepath)

    return df_remodnav


def get_relative_time_gaze_df(gaze_df, gaze_index_timestamp=True):
    if gaze_index_timestamp:
        date_times_relative_in_s = (gaze_df.index - gaze_df.index.min()).total_seconds()
    else:
        date_times_relative_in_s = gaze_df.index.values / 1000000
    date_times_relative_in_s = np.floor(date_times_relative_in_s * 1000) / 1000
    relative_time_gaze_df = pd.DataFrame([date_times_relative_in_s, gaze_df["x"], gaze_df["y"]]).transpose()
    relative_time_gaze_df.columns = ["relative_in_s", "x", "y"]
    relative_time_gaze_df = relative_time_gaze_df.set_index(gaze_df.index)

    return relative_time_gaze_df


def add_blink_and_non_blink(remodnav_df):
    new_events = list()
    for pos in range(len(remodnav_df) - 1):
        time_event_ended_s = remodnav_df.iloc[pos]["onset"] + remodnav_df.iloc[pos]["duration"]
        time_next_event_started_s = remodnav_df.iloc[pos + 1]["onset"]
        duration_without_event_s = time_next_event_started_s - time_event_ended_s
        if duration_without_event_s > 0 and duration_without_event_s > MIN_DURATION_NAN_EVENT:
            new_event = {"onset": time_event_ended_s, "duration": duration_without_event_s}
            if duration_without_event_s <= MAX_DURATION_BLINK_S:
                new_event["label"] = "BLINK"
            elif duration_without_event_s > MAX_DURATION_BLINK_S:
                new_event["label"] = "MISSING_DATA"
            new_events.append(new_event)

    new_events_df = pd.DataFrame(new_events)
    remodnav_df = pd.concat([remodnav_df, new_events_df], axis=0).sort_values(by=['onset']).reset_index()
    remodnav_df = remodnav_df.drop(columns=["index"])

    return remodnav_df


def add_index_and_mean(relative_time_gaze_df, remodnav_df):
    closest_index, mean_x, mean_y = list(), list(), list()
    for index, row in remodnav_df.iterrows():
        closest_index.append(relative_time_gaze_df[relative_time_gaze_df["relative_in_s"] >= row["onset"]].index[0])
        between_mask = (relative_time_gaze_df["relative_in_s"] >= row["onset"]) & (
                relative_time_gaze_df["relative_in_s"] <= (row["onset"] + row["duration"]))
        if row["label"] == "BLINK" or row["label"] == "MISSING_DATA":
            mean_x.append(np.nan)
            mean_y.append(np.nan)
        else:
            mean_x.append(relative_time_gaze_df[between_mask]["x"].mean())
            mean_y.append(relative_time_gaze_df[between_mask]["y"].mean())

    remodnav_df = remodnav_df.set_index(pd.Series(closest_index))
    remodnav_df["mean_x"] = mean_x
    remodnav_df["mean_y"] = mean_y

    return remodnav_df


def get_label_in_dict(label, dict_keys):
    final_lab = [x for x in dict_keys if x in label]
    if len(final_lab) > 0:
        return final_lab[0]
    else:
        return None


def add_to_existing_df_remodnav(gaze_df, remodnav_df):
    event_series = pd.Series([0] * len(gaze_df), index=gaze_df.index)
    event_count = {"FIXA": event_series.copy(), "SAC": event_series.copy(), "PURS": event_series.copy(),
                   "BLINK": event_series.copy()}
    duration_series = pd.to_timedelta(remodnav_df["duration"], unit="s")
    for pos in range(len(remodnav_df) - 1):
        event_label = remodnav_df["label"].iloc[pos]
        matched_label = get_label_in_dict(event_label, event_count.keys())
        if matched_label is not None:
            current_event = event_count[matched_label]
            mask_current_event_timestamps = (current_event.index >= remodnav_df.index[pos]) & (
                    (current_event.index < remodnav_df.index[pos + 1]) |
                    (current_event.index < remodnav_df.index[pos] + duration_series.iloc[pos]))
            current_event[mask_current_event_timestamps] = 1

    df_gaze_remodnav = gaze_df.copy()

    for key in event_count:
        df_gaze_remodnav[key] = event_count[key]

    return df_gaze_remodnav


def resample_for_remodnav(gaze_df, hz):
    gaze_df[["x", "y"]] = gaze_df[["x", "y"]].fillna(np.inf)
    resample_in_ns = round((1 / hz) * 1000000000)
    resampled_gaze_df = gaze_df.resample(str(resample_in_ns) + "N").nearest()

    resampled_gaze_df.replace([np.inf], np.nan, inplace=True)
    gaze_df.replace([np.inf], np.nan, inplace=True)

    return resampled_gaze_df


def create_remodnav_event_files(filepath):
    print("Processing files:", filepath)
    driver = [d for d in DISTANCE_TO_SCREEN_CM_DICT if d in filepath][0]
    filename_without_extension = os.path.split(filepath)[-1].split(".")[0]
    tmp_store_filepath = os.path.join("tmp", filename_without_extension + "_remodnav.tsv")
    filepath_store_event_file = os.path.join(PATH_OUT, filename_without_extension + "_event_remodnav.csv")
    filepath_store_gaze_with_event_file = os.path.join(PATH_OUT, filename_without_extension + "_gaze_remodnav.csv")
    tmp_read_filepath = os.path.join("tmp", "tmp_" + filename_without_extension + ".tsv")

    gaze_df = read_eye_gaze_into_df(filepath, "remodnav", SCREEN_PX_WIDTH, SCREEN_PX_HEIGHT)
    gaze_df = calc_sort_time(gaze_df, False)

    resampled_gaze_df = resample_for_remodnav(gaze_df, EYE_TRACKER_HZ)

    relevant_gaze_df = resampled_gaze_df[["x", "y"]]
    relevant_gaze_df.to_csv(tmp_read_filepath, sep='\t', header=False, index=False)

    remodnav_df = apply_remodnav(tmp_read_filepath, tmp_store_filepath,
                                 DISTANCE_TO_SCREEN_CM_DICT[driver],
                                 SAVGOL_TARGET_SAMPLES, MEDIAN_FILTER_TARGET_LENGTH_IN_S,
                                 VELOCITY_THRESHOLD_START_VELOCITY, PURSUIT_VELOCITY_THRESHOLD,
                                 SACCADE_CONTEXT_WINDOW_LENGTH)

    os.remove(tmp_read_filepath)

    relative_time_gaze_df = get_relative_time_gaze_df(gaze_df)
    remodnav_df = add_blink_and_non_blink(remodnav_df)
    remodnav_df = add_index_and_mean(relative_time_gaze_df, remodnav_df)

    df_gaze_remodnav = add_to_existing_df_remodnav(gaze_df, remodnav_df)

    remodnav_df.to_csv(filepath_store_event_file)
    df_gaze_remodnav.to_csv(filepath_store_gaze_with_event_file)


if __name__ == '__main__':
    filepaths = glob(os.path.join(PATH_IN, GAZE_FILE_PATTERN))
    if len(filepaths) == 0:
        print(PATH_IN, f"has no files with the pattern {GAZE_FILE_PATTERN}")
    print(f"Processing {len(filepaths)} files from {PATH_IN}...")
    Parallel(n_jobs=NUM_CORES, backend='multiprocessing')(
        delayed(create_remodnav_event_files)(f) for f in filepaths)
    print(f"{len(filepaths)} files processed. Output stored in {PATH_OUT}.")
