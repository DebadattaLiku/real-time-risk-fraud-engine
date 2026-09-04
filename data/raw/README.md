# Raw Dataset --- IEEE-CIS Fraud Detection

This directory contains the raw data used by the **Real-Time Risk
Decision & Fraud Intelligence Engine**.

The project uses the **IEEE-CIS Fraud Detection** dataset provided
through the IEEE Computational Intelligence Society fraud detection
competition on Kaggle.

## Dataset Source

**Official Kaggle Competition:**\
https://www.kaggle.com/competitions/ieee-fraud-detection/data

The dataset is subject to the competition's Kaggle rules and terms.

## Dataset Description

The IEEE-CIS Fraud Detection dataset contains anonymized e-commerce
transaction data used to predict whether an online transaction is
fraudulent.

The data is divided into two main groups:

-   **Transaction data** --- transaction-level information such as
    transaction amount, product information, card-related attributes,
    addresses, email domains, and other anonymized variables.
-   **Identity data** --- additional anonymized information related to
    device and identity characteristics for a subset of transactions.

The transaction and identity tables are associated using:

``` text
TransactionID
```

Not every transaction has corresponding identity information.

## Expected Files

After downloading the dataset, this directory should contain:

``` text
data/
└── raw/
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    ├── test_identity.csv
    └── sample_submission.csv
```

### File Descriptions

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `train_transaction.csv`             Training transaction records
                                      containing the `isFraud` target

  `train_identity.csv`                Identity/device information for a
                                      subset of training transactions

  `test_transaction.csv`              Test transaction records

  `test_identity.csv`                 Identity/device information for a
                                      subset of test transactions

  `sample_submission.csv`             Example Kaggle submission format
  -----------------------------------------------------------------------

The official Kaggle dataset contains these five files.

## Dataset Size

The complete Kaggle dataset is approximately **1.35 GB**.

Because the raw CSV files are large, they are **intentionally excluded
from Git version control**.

The repository therefore stores this README instead of redistributing
the raw dataset.

## How to Obtain the Dataset

### Option 1 --- Kaggle Website

Open the official dataset page:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

Download the required files and place the extracted CSV files directly
inside:

``` text
data/raw/
```

### Option 2 --- Kaggle CLI

If the Kaggle CLI is configured, the competition files can be downloaded
using:

``` bash
kaggle competitions download -c ieee-fraud-detection
```

After downloading, extract the files and place them inside:

``` text
data/raw/
```

## Verification

Before running the project, verify that the expected files are present:

``` text
data/raw/train_transaction.csv
data/raw/train_identity.csv
data/raw/test_transaction.csv
data/raw/test_identity.csv
data/raw/sample_submission.csv
```

The training transaction table contains the binary target:

``` text
isFraud
```

where:

``` text
0 → legitimate transaction
1 → fraudulent transaction
```

## Important Reproducibility Note

The raw dataset is an **external dependency** of this project.

It is deliberately not committed to GitHub because of its size and
because the dataset is distributed through Kaggle under its competition
rules.

This README records the authoritative source and expected file structure
so that the complete data pipeline can be reproduced without needing to
remember where the dataset originated.

## Competition and Data Terms

The dataset is provided by Vesta Corporation for the IEEE-CIS Fraud
Detection competition hosted by Kaggle and the IEEE Computational
Intelligence Society. The competition rules state that the competition
data is for permitted non-commercial purposes such as competition
participation, academic research, and education, and impose restrictions
on unauthorized redistribution.

Anyone obtaining the dataset should review and comply with the current
Kaggle competition rules before using or redistributing the data.

## Project Data Flow

``` text
IEEE-CIS Fraud Detection Dataset
                │
                ▼
          data/raw/
                │
                ▼
       Data Preparation
                │
                ▼
        Feature Engineering
                │
                ▼
       Chronological Split
                │
                ▼
       LightGBM + Behavioral
             Features
                │
                ▼
       Risk Decision Engine
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
     APPROVE  REVIEW    BLOCK
```

## Source Citation

Howard, A., Bouchon-Meunier, B., Huang, C., inversion, Lei, J.,
Lynn@Vesta, Marcus2010, & Abbass, H. (2019). **IEEE-CIS Fraud
Detection**. Kaggle.

Official competition page:

https://www.kaggle.com/competitions/ieee-fraud-detection

Official data page:

https://www.kaggle.com/competitions/ieee-fraud-detection/data
