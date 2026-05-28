# Breast Cancer Classification Using Machine Learning

## Overview

This project implements a supervised machine learning pipeline for classifying breast cancer samples as benign or malignant using diagnostic feature data.

The objective is to build a reproducible baseline workflow covering data loading, exploratory analysis, model training, evaluation, and interpretation.

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic Dataset available through `scikit-learn`.

The dataset contains numerical features computed from digitized images of fine needle aspirate samples of breast masses. Each sample is labelled as either benign or malignant.

## Project Structure

```text
breast-cancer-classification/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_exploration_and_baseline.ipynb
├── src/
│   ├── load_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── metrics.json
└── report/
    └── project_report.pdf