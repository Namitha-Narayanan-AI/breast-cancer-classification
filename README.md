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
├── .gitignore
│
├── notebooks/
│   └── 01_exploration_and_baseline.ipynb
│
├── src/
│   ├── load_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── results/
│   ├── baseline_metrics.json
│   ├── class_distribution.png
│   └── confusion_matrix.png
│
└── report/
    └── project_report.pdf
```

## Methods

The workflow includes:

1. Loading the Breast Cancer Wisconsin Diagnostic Dataset
2. Converting the dataset into a structured pandas DataFrame
3. Performing exploratory data analysis
4. Splitting the dataset into training and test sets using a stratified 80/20 split
5. Scaling numerical features using `StandardScaler`
6. Training a Logistic Regression baseline model
7. Evaluating the model using classification metrics
8. Saving metrics and visual outputs to the `results/` directory

The baseline model is implemented using a `scikit-learn` pipeline:

```text
StandardScaler → LogisticRegression
```

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

For a medical classification task, recall and false negative analysis are important because false negatives may represent malignant cases incorrectly classified as benign.

## Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline training pipeline:

```bash
python src/train_model.py
```

Print the saved evaluation metrics:

```bash
python src/evaluate_model.py
```

Run a single-sample prediction example:

```bash
python src/predict.py
```

## Notebook

The exploratory analysis and baseline modelling notebook is available at:

```text
notebooks/01_exploration_and_baseline.ipynb
```

The notebook includes dataset inspection, class distribution analysis, feature summary, baseline model training, and confusion matrix visualization.

## Results

The baseline Logistic Regression model is trained and evaluated on a stratified train-test split.

Generated outputs are stored in the `results/` directory:

```text
results/
├── baseline_metrics.json
├── class_distribution.png
└── confusion_matrix.png
```

The exact metric values are available in:

```text
results/baseline_metrics.json
```

The class distribution plot shows the balance between benign and malignant samples. The confusion matrix provides a summary of correct and incorrect predictions across both classes.

## Reproducibility

The project uses a fixed random seed during train-test splitting:

```python
random_state=42
```

The dataset is loaded directly from `scikit-learn`, so no external raw data download is required.

## Clinical Relevance and Limitations

This project demonstrates a baseline machine learning workflow for breast cancer classification using structured diagnostic features.

The dataset is derived from digitized fine needle aspirate samples, but this project does not use raw medical images, radiology scans, genomic data, or clinical patient records. Therefore, the results should be interpreted as a technical baseline experiment rather than a clinically deployable diagnostic system.

Further validation on larger, diverse, and clinically representative datasets would be required before considering any real-world medical use.

## Future Work

Possible extensions include:

- Cross-validation and hyperparameter tuning
- ROC curve visualization and false negative analysis
- SHAP-based feature interpretation
- Comparison with additional baseline models
- Extension to breast ultrasound, mammography, or MRI datasets
- Development of a radiomics-focused classification workflow
## Report

A short project report is available at:

```text
report/project_report.pdf
```

## License

This project is for educational and research portfolio purposes.