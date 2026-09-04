# Phase 0 EDA Summary — Real-Time Risk Decision & Fraud Intelligence Engine

## Dataset availability status: **AVAILABLE**

This report reflects a real run against `data/raw/train_transaction.csv` from the IEEE-CIS Fraud Detection dataset. All figures below are measured, not illustrative.

## A. Dataset structure

- Rows: **590,540**
- Columns: **394**
- Duplicate `TransactionID` values: **0**
- `TransactionID` unique: **True**
- Fraud class counts: **{0: 569877, 1: 20663}**
- Overall fraud rate: **3.4990%**

## B. Temporal structure (`TransactionDT`)

- Min: **86,400** seconds
- Max: **15,811,131** seconds
- Range: **15,724,731** seconds (~**182.0** days)
- Rows already in `TransactionDT` order in the raw file: **True**
- Total rows: **590,540**
- Proposed **train** split: rows **0 – 413,377** (413,378 rows, 70.00%), `TransactionDT` up to **10,437,996**
- Proposed **validation** split: rows **413,378 – 501,958** (88,581 rows, 15.00%), `TransactionDT` from **10,437,996** to **13,151,840**
- Proposed **test** split: rows **501,959 – 590,539** (88,581 rows, 15.00%), `TransactionDT` from **13,151,840** onward

(Row boundaries above are positions in the `TransactionDT`-sorted order, not raw file row numbers — the raw file already happens to be close to time-ordered, but sorting is applied explicitly rather than assumed. No random shuffling was used at any point.)

Fraud rate and transaction volume per time bin (20 equal-width bins across the full range):

| Bin | Transaction volume | Fraud rate |
|---|---|---|
| (86399.999, 872636.55] | 36,022 | 2.8566% |
| (872636.55, 1658873.1] | 40,188 | 2.5281% |
| (1658873.1, 2445109.65] | 48,022 | 2.0303% |
| (2445109.65, 3231346.2] | 31,488 | 3.6935% |
| (3231346.2, 4017582.75] | 25,876 | 3.9457% |
| (4017582.75, 4803819.3] | 26,324 | 4.0913% |
| (4803819.3, 5590055.85] | 28,099 | 4.4521% |
| (5590055.85, 6376292.4] | 27,884 | 4.4219% |
| (6376292.4, 7162528.95] | 26,873 | 3.7063% |
| (7162528.95, 7948765.5] | 29,151 | 3.8455% |
| (7948765.5, 8735002.05] | 34,084 | 3.2038% |
| (8735002.05, 9521238.6] | 26,804 | 3.8688% |
| (9521238.6, 10307475.15] | 27,342 | 4.7765% |
| (10307475.15, 11093711.7] | 29,968 | 3.4937% |
| (11093711.7, 11879948.25] | 23,592 | 3.8403% |
| (11879948.25, 12666184.8] | 24,186 | 3.4359% |
| (12666184.8, 13452421.35] | 27,782 | 3.1675% |
| (13452421.35, 14238657.9] | 26,319 | 2.9522% |
| (14238657.9, 15024894.45] | 26,796 | 3.5229% |
| (15024894.45, 15811131.0] | 23,740 | 4.0480% |

## C. Missingness by column group

| Group | # columns | Mean missing % | Max missing % | Min missing % |
|---|---|---|---|---|
| card | 6 | 0.51% | 1.51% | 0.00% |
| addr | 2 | 11.13% | 11.13% | 11.13% |
| D | 15 | 58.15% | 93.41% | 0.21% |
| C | 14 | 0.00% | 0.00% | 0.00% |
| V | 339 | 43.04% | 86.12% | 0.00% |

**Top-5 most-missing columns in `card`:**
- `card2`: 1.51%
- `card5`: 0.72%
- `card4`: 0.27%
- `card6`: 0.27%
- `card3`: 0.27%

**Top-5 most-missing columns in `addr`:**
- `addr1`: 11.13%
- `addr2`: 11.13%

**Top-5 most-missing columns in `D`:**
- `D7`: 93.41%
- `D13`: 89.51%
- `D14`: 89.47%
- `D12`: 89.04%
- `D6`: 87.61%

**Top-5 most-missing columns in `C`:**
- `C1`: 0.00%
- `C2`: 0.00%
- `C3`: 0.00%
- `C4`: 0.00%
- `C5`: 0.00%

**Top-5 most-missing columns in `V`:**
- `V146`: 86.12%
- `V149`: 86.12%
- `V158`: 86.12%
- `V148`: 86.12%
- `V147`: 86.12%

## D. Candidate pseudo-entity key analysis

isFraud was never used to construct these keys. Rows with a missing value in any constituent column are excluded from cardinality/txn-per-entity statistics (see `pseudo_entity_analysis` docstring) and reported separately below, so missingness cannot silently create false entity groups.

### Key: `card1`

- Rows total: **590,540**
- Rows with a missing key component: **0** (0.00%)
- Rows with complete key: **590,540**
- Cardinality (distinct pseudo-entities): **13,553**
- Median transactions/entity: **4.0**
- Mean transactions/entity: **43.57**
- Max transactions/entity: **14,932**
- % of entities with exactly 1 transaction: **25.41%**

### Key: `card1 + card2`

- Rows total: **590,540**
- Rows with a missing key component: **8,933** (1.51%)
- Rows with complete key: **581,607**
- Cardinality (distinct pseudo-entities): **13,490**
- Median transactions/entity: **4.0**
- Mean transactions/entity: **43.11**
- Max transactions/entity: **14,891**
- % of entities with exactly 1 transaction: **25.42%**

### Key: `card1 + card2 + addr1`

- Rows total: **590,540**
- Rows with a missing key component: **74,115** (12.55%)
- Rows with complete key: **516,425**
- Cardinality (distinct pseudo-entities): **37,280**
- Median transactions/entity: **2.0**
- Mean transactions/entity: **13.85**
- Max transactions/entity: **5,866**
- % of entities with exactly 1 transaction: **39.51%**

### Key: `card1 + card2 + card3 + card5 + addr1 + addr2`

- Rows total: **590,540**
- Rows with a missing key component: **75,885** (12.85%)
- Rows with complete key: **514,655**
- Cardinality (distinct pseudo-entities): **38,197**
- Median transactions/entity: **2.0**
- Mean transactions/entity: **13.47**
- Max transactions/entity: **5,862**
- % of entities with exactly 1 transaction: **39.50%**

### Recommendation

`card1` alone is recommended as the Phase 1 starting point for behavioral aggregation: it has zero missing values (no rows excluded), the lowest cardinality-to-coverage tradeoff loss, and gives every transaction a usable pseudo-entity. Adding `card2` barely changes cardinality (13,553 → 13,490) while excluding 1.51% of rows, suggesting `card1` is already close to saturating the identity signal available in the `card*` fields alone. Adding `addr1`/`addr2` roughly triples cardinality (13.5K → ~37-38K) and excludes ~12.5-12.9% of rows — this likely fragments genuine repeat entities into multiple pseudo-entities (e.g. the same card used with a missing or differently-imputed address) rather than adding real identity resolution power, which is the opposite of what a behavioral feature needs. This is a judgment call from the measured tradeoffs above, not a validated ground truth — there is no real customer ID in this dataset to check against.

**Limitations (apply to all candidates above):** none of these keys are verified customer identities — `card1`/`card2` are Vesta's anonymized/hashed card identifiers and may not 1:1 map to a single physical card or person (shared cards, re-issued cards, or hash collisions are all possible and unverifiable from this data). Cardinality and transactions-per-entity are therefore proxies for behavioral grouping, not proof of identity. Any velocity/frequency feature built on these keys must be documented as resting on this assumption.

## Integrity tests

| Test | Result |
|---|---|
| transaction_id_unique | PASS |
| no_row_overlap_train_val | PASS |
| no_row_overlap_val_test | PASS |
| no_row_overlap_train_test | PASS |
| max_train_dt_lte_min_val_dt | PASS |
| max_val_dt_lte_min_test_dt | PASS |

## Figures

- `reports/figures/transaction_volume_over_time.png`
- `reports/figures/fraud_rate_over_time.png`
- `reports/figures/missingness_by_group.png`
