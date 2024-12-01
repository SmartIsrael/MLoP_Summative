# Data Directory

This directory contains the datasets used in the project.

## Structure

- `train/`: Contains the training dataset.
- `test/`: Contains the test dataset.

## Data Description

The datasets in this directory are related to heart failure prediction. They include various clinical features such as age, anaemia status, creatinine levels, and other relevant medical indicators.

## About the data:

- age: Age of the patient
- anaemia: If the patient had the haemoglobin below the normal range
- creatinine_phosphokinase: The level of the creatine phosphokinase in the blood in mcg/L
- diabetes: If the patient was diabetic
- ejection_fraction: Ejection fraction is a measurement of how much blood the left ventricle pumps out with each contraction
- high_blood_pressure: If the patient had hypertension
- platelets: Platelet count of blood in kiloplatelets/mL
- serum_creatinine: The level of serum creatinine in the blood in mg/dL
- serum_sodium: The level of serum sodium in the blood in mEq/L
- sex: The sex of the patient
- smoking: If the patient smokes actively or ever did in past
- time: It is the time of the patient's follow-up visit for the disease in months
- DEATH_EVENT: If the patient deceased during the follow-up period

## Usage

The data in this directory is used by the preprocessing scripts and the main notebook for model training and evaluation. To use the data:

1. Ensure you're in the project's root directory.
2. Load the data in your Python script or notebook:

```python
import pandas as pd

train_data = pd.read_csv('data/train/heart_failure_clinical_records_dataset.csv')
test_data = pd.read_csv('data/test/heart_failure_clinical_records_dataset.csv')
```

Note: Always handle medical data with care and ensure compliance with relevant data protection regulations.
