# CHI 2023 paper – Leveraging driver vehicle and environment interaction: Machine learning using driver monitoring cameras to detect drunk driving

This page contains our feature engineering pipeline source code for our manuscript submitted to CHI 2023:
> Leveraging driver vehicle and environment interaction: Machine learning using driver monitoring cameras to detect drunk driving

## Content
This repo consists of three major parts: (i) a command line tool to record the eye tracking data from a [Tobii Nano Pro](https://www.tobiipro.com/product-listing/nano), (ii) a tool to calculate eye event data, and (iii) a tool to calculate features from the eye tracking data.
We will describe in the following on how to get started with this code in more detail.

> **Prerequisites**: We recommend to use Python 3.8 and to install dependencies via `pip install -U -r requirements.txt`

- `tobii_nano_pro_recorder`: A dedicated README file in the folder explains on how to use the command line tool to record Tobii Nano Pro data.
- `eye_event_classification`: We use the [REMoDNaV](https://github.com/psychoinformatics-de/remodnav) algorithm to annotate the collected eye tracking data with additional events. In `config/remodnav_config.json` are run-specific parameters defined. In particular, we calibrated the REMoDNaV on self-annotated eye tracking data to the current parameter settings.
- `eye_feature_engineering`: Our custom feature engineering pipeline to create features for the prediction of drunk drivers. Several parameters can be changed in `config/feature_engineering_config.json`
- `prediction`: Here, we provide the output of our main analysis for our paper.
- `examples`: In this folder, we provide a simple dataset that we recorded with the Tobii Nano Pro to test our pipeline. `eye_event_classification` and `eye_feature_engineering` can be executed with this sample data.

## Citation

**Please cite our paper in any published work that uses any of these resources.**

BiBTeX:
```
TBA
```

ACM Ref Citation:

**TBA**

## Contact
Please contact [Kevin Koch](kevin.koch@unisg.ch) or [Martin Maritsch](mmaritsch@ethz.ch) for questions.

## 
