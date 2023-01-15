import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------
# LOADING EYE FILES

def read_eye_gaze_into_df(gaze_file: str, data_type: str, screen_px_width=None, screen_px_height=None) -> pd.DataFrame:
    """
    Function to read in eye gaze data CSV file generated by Tobii Eye Tracker,
    remove timezone and shift x coordinates
    gaze_file: (str) name of gaze files
    :return: (pd.DataFrame) data loaded into dataframe
    """
    value_columns = ["x", "y", "head_x", "head_y", "head_z"]
    if data_type == "remodnav":
        gaze_df = pd.read_csv(gaze_file, sep=',', index_col=0, header=0, skiprows=None, parse_dates=[0])
        gaze_df["x"] = gaze_df[["left_gaze_point_on_display_area_1", "right_gaze_point_on_display_area_1"]].mean(axis=1)
        gaze_df["y"] = gaze_df[["left_gaze_point_on_display_area_2", "right_gaze_point_on_display_area_2"]].mean(axis=1)
        gaze_df["head_x"] = gaze_df[
            ["left_gaze_origin_in_user_coordinate_system_1", "right_gaze_origin_in_user_coordinate_system_1"]].mean(
            axis=1)
        gaze_df["head_y"] = gaze_df[
            ["left_gaze_origin_in_user_coordinate_system_2", "right_gaze_origin_in_user_coordinate_system_2"]].mean(
            axis=1)
        gaze_df["head_z"] = gaze_df[
            ["left_gaze_origin_in_user_coordinate_system_3", "right_gaze_origin_in_user_coordinate_system_3"]].mean(
            axis=1)
        gaze_df = gaze_df[["device_time_stamp"] + value_columns]

        gaze_df['x'] = gaze_df['x'] * screen_px_height
        gaze_df['y'] = gaze_df['y'] * screen_px_width

        # sort indexes
        gaze_df.sort_index(inplace=True)
        gaze_df.columns = ['relative_time'] + value_columns

        # ignore +1 timezone
        gaze_df = gaze_df.tz_localize(None)

    elif data_type == "feature":
        gaze_df = pd.read_csv(gaze_file, header=0, skiprows=None, parse_dates=[0], index_col=0)

    return gaze_df


# recalculate timestamps for eye-tracking with initial_absolute_timestamp + delta_relative_timestamp
def calc_sort_time(data, drop_relative_time=True, leading_data=None, timestamp_unit="us"):
    """
    Function to recalculate timestamps for eye-tracking with initial_absolute_timestamp + delta_relative_timestamp
    data: (pd.DataFrame)  data with timestamps as indexes and column relative_time
    :return: (pd.DataFrame) data with recalculated time indexes
    """
    # minimum of timestamps used as starting time
    if leading_data is None:
        initial_timestamp = data.index.min()

        # find corresponding offset difference / relative time
        min_diffs = data.at[initial_timestamp, 'relative_time']
    else:
        initial_timestamp = leading_data.index.min()
        min_diffs = leading_data.at[initial_timestamp, 'relative_time']

    initial_diff = min_diffs.min()  # if multiple same datetimes/indexes

    # convert datetime[0] to relative time format [ms till 1970]
    initial_timestamp_ms = (initial_timestamp - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,
                                                                                                        timestamp_unit)

    # datetime = index = start_time + relative_time - relative_time[0]
    data.index = pd.to_datetime(initial_timestamp_ms + data['relative_time'] - initial_diff, unit=timestamp_unit)
    if drop_relative_time:
        data.drop(['relative_time'], axis=1, inplace=True)
    data.index.name = 'datetime'

    # sort and remove duplicates
    data.sort_index(inplace=True)
    data = data.loc[~data.index.duplicated(keep='first')]

    return data


def read_remodav_event_into_df(event_file: str, event_types: [str]) -> pd.DataFrame:
    """
    Function to read in event gaze data CSV file generated by REMODNAV algorithm,
    filter for relevent events, removes timezone and shift x coordinates
    event_file: (str) name of event file
    event_types: (str) list of events to keep
    :return: (pd.DataFrame) data loaded into dataframe
    """
    event_df = pd.read_csv(event_file, index_col=0, parse_dates=[0])
    event_df = event_df[event_df["label"].isin(event_types)]
    if "ISAC" in event_types:
        event_df["label"] = event_df["label"].str.replace('ISAC', 'SACC')

    event_df = event_df.tz_localize(None)
    return event_df