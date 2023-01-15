from tracker import Tracker
import datetime
import os
import sys

print("Tobii Eyetracker Pro SDK Recorder v0.1")

print("Eye tracker calibrated? Confirm with 'y':")
is_calibrated = input()
if is_calibrated != "y":
    print("Please calibrate.")
    sys.exit()

if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
    print("Please specify the output directory as first and only argument.")
    sys.exit()

now = datetime.datetime.now()
filepath = os.path.join(sys.argv[1], now.strftime("%Y-%m-%d--%H-%M-%S_") + "gazepoints.csv")
print("Filepath:", filepath)

eye_tracker = Tracker(filepath)

eye_tracker.subscribe()

should_continue = True
print()
while should_continue:
    print("To stop the script, enter 'y' and confirm with enter:")
    input_line = input()
    if input_line == "y":
        should_continue = False

eye_tracker.unsubscribe()

eye_tracker.write_to_file()

print("File is written to", filepath)
