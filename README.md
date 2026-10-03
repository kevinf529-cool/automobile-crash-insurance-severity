# automobile-crash-insurance-severity
Analysis of fatal crash outcomes and private-auto insurance claim severity using statistical analysis and machine learning.

# Automobile Crash Risk & Insurance Claim Severity

Analysis of U.S. fatal crash outcomes and private-auto insurance claim severity using statistical analysis and machine learning.

## Project Overview

Automobile risk can be examined from both a public-safety perspective and an insurance perspective. This project analyzes two independent datasets to explore those different dimensions of risk.

The first analysis uses 2024 data from the National Highway Traffic Safety Administration's Fatality Analysis Reporting System (FARS) to examine characteristics associated with fatal outcomes among drivers and passengers represented in fatal crashes.

The second analysis uses private-auto insurance claim data from CASdatasets to examine factors associated with higher claim payments and determine whether the available variables can reliably predict individual claim severity.

Because the datasets represent different populations and outcomes, they were analyzed independently and were never joined.

## Research Questions

1. Which occupant, vehicle, and crash characteristics are associated with fatal outcomes among drivers and passengers represented in 2024 FARS fatal crashes?

2. Which available driver and insurance characteristics are associated with higher private-auto claim payments?

## Datasets

### NHTSA FARS

The FARS analysis uses the 2024 Person, Vehicle, and Accident files from the National Highway Traffic Safety Administration.

After selecting drivers and passengers and preparing the injury outcome variable, the modeling population contained **77,285 occupant records**:

- **30,658 fatal outcomes (39.67%)**
- **46,627 nonfatal outcomes (60.33%)**

FARS is a census of fatal crashes rather than all U.S. crashes. Therefore, the results describe patterns within the FARS fatal-crash population and should not be interpreted as general crash or fatality probabilities for all drivers.

### CAS Private-Auto Claims

The insurance analysis uses the `usprivautoclaim` dataset from the CASdatasets package.

The dataset contains **6,773 observations** and five variables:

- `STATE`
- `CLASS`
- `GENDER`
- `AGE`
- `PAID`

The `PAID` variable represents the claim-payment amount and serves as the target for the claim-severity models.

## Data Preparation

The FARS Person, Vehicle, and Accident files were connected using the `ST_CASE` crash identifier. Vehicle information was linked using crash and vehicle identifiers, while duplicate checks were performed using crash, vehicle, and person keys.

An important data-quality issue involved FARS numeric sentinel codes. Several fields initially appeared to contain no missing values because unavailable information was stored using numeric codes rather than standard `NaN` values.

Examples included:

- `AGE`: 998 and 999
- `MOD_YEAR`: 9998 and 9999
- `VSPD_LIM`: 98 and 99
- `HOUR`: 99

These values were identified and converted to missing values where appropriate. Valid values, such as a speed limit of zero, were retained.

For the insurance dataset, claim-payment distributions and potential outliers were examined before modeling. High-dollar claims were retained because they represent meaningful insurance losses rather than automatically being treated as data errors.

## Exploratory Analysis

The FARS analysis showed substantial differences in fatal-outcome percentages across several characteristics.

### Restraint Classification

- **Restraint Used:** 23.4% fatal outcome
- **None Used / Not Applicable:** 72.5% fatal outcome

### Vehicle Body Type

Selected fatal-outcome percentages included:

- **Motorcycle:** 90.4%
- **Passenger Car:** 42.8%
- **Pickup:** 32.4%
- **SUV/Utility:** 31.4%
- **Bus:** 4.2%

### Speed Relation

- **Speed Related:** 59.6% fatal outcome
- **Not Speed Related:** 33.8% fatal outcome

These percentages describe occupants represented within FARS fatal crashes and are not population-level fatality probabilities.

The insurance claim data were strongly right-skewed:

- **Mean claim payment:** approximately $1,853
- **Median claim payment:** approximately $1,002

Average claim payments also varied across insurance classes, although these descriptive differences did not necessarily translate into strong individual-level predictive performance.

## Statistical Analysis

Categorical relationships with fatal outcome were evaluated using chi-square tests and Cramér's V.

Some of the strongest observed associations included:

| Variable | Cramér's V |
|---|---:|
| Restraint Classification | 0.4512 |
| Ejection | 0.4387 |
| Harmful Event | 0.4240 |
| Rollover | 0.3908 |
| Vehicle Body Type | 0.3699 |

These relationships are descriptive associations and should not be interpreted as evidence of causation.

Several crash-event variables with strong descriptive relationships were excluded from predictive modeling because they could introduce information that occurs during or after the crash event and create leakage concerns.

## Predictive Modeling

### FARS Fatal-Outcome Models

Logistic Regression and Random Forest classifiers were evaluated using **five-fold crash-level cross-validation**.

Occupants from the same crash were kept within the same fold to reduce information leakage between training and validation data.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 75.07% | 74.04% |
| Precision | 73.94% | 66.85% |
| Recall | 57.37% | 68.61% |
| F1 Score | 0.6461 | 0.6771 |
| ROC-AUC | 0.8054 | 0.8058 |

Both models achieved ROC-AUC values near **0.81**, indicating similar overall discrimination.

Logistic Regression produced higher precision, while Random Forest produced higher recall and F1 score. The results demonstrate a model tradeoff rather than a single overall winner.

### Insurance Claim-Severity Models

Linear Regression and Random Forest Regression were used to predict the `PAID` claim amount.

| Model | MAE | R² |
|---|---:|---:|
| Linear Regression | ~$1,522.92 | -0.0036 |
| Random Forest | ~$1,537.86 | -0.0197 |

Both models produced negative R² values, indicating that the available variables did not explain enough variation in individual claim payments to support reliable claim-severity prediction.

Additional experiments using a log-transformed target reduced dollar MAE after back-transformation but did not resolve the weak explanatory performance.

## Key Findings

- Fatal outcomes within the FARS population varied substantially across restraint classification, vehicle body type, speed relation, and other crash characteristics.
- Logistic Regression and Random Forest achieved similar ROC-AUC values near 0.81 but demonstrated different precision and recall tradeoffs.
- Private-auto claim payments were strongly right-skewed and varied descriptively across insurance classes.
- `STATE`, `CLASS`, `GENDER`, and `AGE` alone were not sufficient for reliable individual claim-severity prediction.
- Descriptive relationships do not automatically guarantee strong predictive performance.

## Limitations & Ethical Considerations

Several limitations are important when interpreting the results:

- FARS contains fatal crashes rather than all U.S. crashes.
- FARS results cannot be used to estimate the average driver's probability of crashing or dying in a crash.
- Observed statistical relationships represent association rather than causation.
- Numeric sentinel codes require careful identification during data preparation.
- Crash-event variables were excluded from predictive modeling when they presented potential leakage concerns.
- Small or ambiguous categories require cautious interpretation.
- The insurance dataset contains a limited number of predictors and lacks richer claim-level information.
- Predictive models should support analysis rather than independently automate high-stakes decisions.

## Tools & Technologies

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- Jupyter Notebook
- Logistic Regression
- Random Forest Classification
- Linear Regression
- Random Forest Regression
- Cross-Validation
- Chi-Square Analysis
- Cramér's V

## Repository Structure

```text
automobile-crash-insurance-severity/
│
├── README.md
└── automobile_crash_insurance_analysis.ipynb
