import threading
import datetime
import tobii_research as tr
import pandas as pd
import os
import sys

THRESHOLD_TO_WRITE = 10 * 60  # save every 10 sec with 60Hz frequency
COLUMNS_2D = ["left_gaze_point_on_display_area", "right_gaze_point_on_display_area"]
COLUMNS_3D = ["left_gaze_point_in_user_coordinate_system",
              "left_gaze_origin_in_user_coordinate_system",
              "left_gaze_origin_in_trackbox_coordinate_system",
              "right_gaze_point_in_user_coordinate_system",
              "right_gaze_origin_in_user_coordinate_system",
              "right_gaze_origin_in_trackbox_coordinate_system"]


class AsyncWrite(threading.Thread):

    def __init__(self, data, filepath):
        threading.Thread.__init__(self)
        self.data = data
        self.filepath = filepath

    def run(self):
        data_df = pd.DataFrame(self.data)

        for val in COLUMNS_2D:
            data_df[[val + "_1", val + "_2"]] = pd.DataFrame(data_df[val].tolist())

        for val in COLUMNS_3D:
            data_df[[val + "_1", val + "_2", val + "_3"]] = pd.DataFrame(data_df[val].tolist())

        data_df = data_df.set_index(data_df["datetime"])
        data_df = data_df.drop(["datetime"] + COLUMNS_2D + COLUMNS_3D, axis=1)
        if not os.path.isfile(self.filepath):
            data_df.to_csv(self.filepath)
        else:
            data_df.to_csv(self.filepath, mode="a", header=False)


class Tracker:

    def __init__(self, filepath="test.csv"):
        self.gaze_data_list = list()
        self.filepath = filepath
        found_eyetrackers = tr.find_all_eyetrackers()
        if len(found_eyetrackers) == 0:
            print("Please connect the eye tracker")
            sys.exit()
        self.eye_tracker = found_eyetrackers[0]
        print("Address: " + self.eye_tracker.address)
        print("Model: " + self.eye_tracker.model)
        print("Name (It's OK if this is empty): " + self.eye_tracker.device_name)
        print("Serial number: " + self.eye_tracker.serial_number)

    def write_to_file(self):
        background = AsyncWrite(self.gaze_data_list, self.filepath)
        self.gaze_data_list = list()
        background.start()

    def gaze_data_callback(self, gaze_data):
        gaze_data["datetime"] = datetime.datetime.now(datetime.timezone.utc)
        self.gaze_data_list.append(gaze_data)
        if len(self.gaze_data_list) > THRESHOLD_TO_WRITE:
            self.write_to_file()

    def subscribe(self):
        self.eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)

    def unsubscribe(self):
        self.eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
