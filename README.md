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
@inproceedings{10.1145/3411764.3446865,
author = {Koch, Kevin and Maritsch, Martin and van Weenen, Eva and Feuerriegel, Stefan and Pfäffli, Matthias and Fleisch, Elgar and Weinmann, Wolfgang and Wortmann, Felix},
title = {Leveraging driver vehicle and environment interaction: Machine learning using driver monitoring cameras to detect drunk driving},
year = {2023},
isbn = {97814503942152304},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3544548.3580975},
doi = {10.1145/3544548.3580975},
booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
articleno = {293},
numpages = {32},
keywords = {Field study, Mindfulness, In-vehicle interventions, Music, Natural driving, Psychology, Well-being},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

ACM Ref Citation:

Kevin Koch, Martin Maritsch, Eva van Weenen, Stefan Feuerriegel, Matthias Pfäffli, Elgar Fleisch, Wolfgang Weinmann, and Felix Wortmann. 2023. Leveraging driver vehicle and environment interaction: Machine learning using driver monitoring cameras to detect drunk driving. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI ’23), April 23–28, 2023, Hamburg, Germany.* ACM, New York, NY, USA, 32 pages.
https://doi.org/10.1145/3544548.3580975

## Contact
Please contact [Kevin Koch](kevin.koch@unisg.ch) or [Martin Maritsch](mmaritsch@ethz.ch) for questions.

## 
