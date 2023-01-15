# Tobii Pro Nano Recorder (CLI tool)

Used to record the data of the Tobii Pro Nano with the [Tobii Pro SDK (see for more information on the collected data and data collection)](https://www.tobiipro.com/product-listing/tobii-pro-sdk/#Download).

## Prerequisites
- Use the [Eye Tracker Manager]( http://developer.tobiipro.com/eyetrackermanager.html) to calibrate the eyetracker
- To create an executable file with all dependecies, install:
`pyinstaller record_tracker.py --onefile --hidden-import="tobiiresearch.implementation.Errors" --hidden-import="tobiiresearch.implementation.ExternalSignalData" --hidden-import="tobiiresearch.implementation.EyeImageData" --hidden-import="tobiiresearch.implementation.EyeTracker" --hidden-import="tobiiresearch.implementation.GazeData" --hidden-import="tobiiresearch.implementation.License" --hidden-import="tobiiresearch.implementation._LogEntry" --hidden-import="tobiiresearch.implementation.Notifications" --hidden-import="tobiiresearch.implementation.ScreenBasedCalibration" --hidden-import="tobiiresearch.implementation.StreamErrorData" --hidden-import="tobiiresearch.implementation.TimeSynchronizationData" --hidden-import="tobiiresearch.implementation.TrackBox" --hidden-import="tobiiresearch.implementation.HMDLensConfiguration" --hidden-import="tobiiresearch.implementation.UserPositionGuide" --hidden-import="tobiiresearch.implementation.Calibration" --hidden-import="tobiiresearch.implementation.StreamErrorData" --hidden-import="tobiiresearch.implementation.TimeSynchronizationData" --hidden-import="tobiiresearch.implementation.TrackBox" --hidden-import="tobiiresearch.implementation.HMDGazeData" --hidden-import="tobiiresearch.implementation.ScreenBasedCalibration" --hidden-import="tobiiresearch.implementation.HMDBasedCalibration" --hidden-import="tobiiresearch.implementation.ScreenBasedMonocularCalibration"
`
- The last command creates an executable file that can you can use via the command line.

## How to use it
Start via command line the  `record_tracker.exe` and define as `arg1` the path for storing the .csv gaze data.