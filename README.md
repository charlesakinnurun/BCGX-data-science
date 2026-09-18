# BCG Data Science — PowerCo Customer Churn Prediction & Retention Analysis

A complete end-to-end data science case study from the **BCG X Data Science
Virtual Experience Programme**: predict which energy customers are about to churn, test the
client's hypothesis that *churn is driven by price sensitivity*, and estimate
the value of a retention discount.

Every stage of a real consulting engagement is covered — framing the business
problem, exploratory analysis, feature engineering, machine learning modelling
and a final recommendation to the client.

![status](https://img.shields.io/badge/status-complete-4c1)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB)
![tasks](https://img.shields.io/badge/tasks-1%20%E2%80%93%204-0061ff)
![model](https://img.shields.io/badge/model-Scikit--learn%20Random%20Forest-8A8A8A)

---

## Project Overview

The client is a large European energy retailer (**PowerCo**). Around 10% of
its customers churn within any 3-month period, and the sales division believes
the main driver is **price**: that a price increase causes customers to leave.
As the data scientist on the engagement, you must verify or refute this
hypothesis and, if churn *can* be predicted, show the financial value of
acting on those predictions.

The engagement is delivered in four tasks:

| Task | Focus | Deliverable |
|------|-------|-------------|
| 1 | Business understanding & hypothesis framing | Client email |
| 2 | Exploratory data analysis (EDA) | Analysis notebook + cleaned data |
| 3 | Feature engineering & churn modelling | Model + prediction file |
| 4 | Findings & recommendation | Executive summary |

**Headline results**

- **The price hypothesis does not hold.** EDA and model feature-importance
  both show that price variables barely correlate with churn — consumption,
  power and margin are far stronger drivers.
- A tuned **Random Forest** predicts retention with ~90.4% accuracy, but its
  *churn* recall is only ~4.8% because the dataset is strongly imbalanced
  (roughly 9:1 in favour of non-churners).
- A simulated **20% retention discount** for predicted churners adds
  **~$3.2k** of retained revenue; focusing the discount on high-value
  customers alone does not meaningfully change that figure.

<p align="center">
  <img src="assets/output7.png" alt="Correlation heatmap of engineered features vs churn" width="420"/>
</p>
*Correlation heatmap of the engineered features against `churn` (Task 3).
Price columns are clustered but show almost no association with churn.*

---

## Project Structure

```
BCG-data-science/
├── Task 1 Business Understanding and Hypothesis Framing/
│   ├── 1. Task 1 Video Instruction.mp4
│   ├── 2. Background Information .pdf
│   ├── 3. Task Brief.pdf
│   ├── 4. Task Resource - Problem Context Document.pdf
│   └── Solution/
│       └── Task_1_email.pdf            # hypothesis framed for the client
├── Task 2 Exploratory Data Analysis/
│   ├── 1. Task 2 Video Instruction.mp4
│   ├── 2. Background Information .pdf
│   ├── 3. Task Brief.pdf
│   ├── Solution/
│   │   ├── BCG_TASK2.ipynb             # full EDA notebook
│   │   ├── BCG_TASK2.pdf               # exported EDA report
│   │   └── clean_data_after_eda.csv    # cleaned dataset for Task 3
│   └── Task Resources/
│       ├── 4.1 Client Data.csv         # 14,606 × 26 client records
│       ├── 4.2 Price Data.csv          # 193,002 × 8 monthly prices (2015)
│       ├── 4.3 Data Description.docx
│       ├── 4.4 Task Description.pptx
│       └── 4.5 eda_starter.ipynb       # starting template
├── Task 3 Feature Engineering and Modeling/
│   ├── 1. Task 3 Video Instruction.mp4
│   ├── 2. Background Information .pdf
│   ├── 3. Task Brief.pdf
│   ├── Solution/
│   │   ├── feature_engineering.ipynb   # features → model → discount analysis
│   │   └── data/
│   │       └── predicted_data.csv      # per-customer churn probabilities
│   └── Task Resources/
│       └── 4.1 feature_engineering.ipynb
├── Task 4 FIndings and Recommendation/
│   ├── 1. Task 4 Video Instruction.mp4
│   ├── 2. Background Information .pdf
│   ├── 3. Task Brief.pdf
│   ├── Solutions/
│   │   └── Executive Summary.pdf       # final client recommendation
│   └── Task Resources/
│       ├── 4.1 Template Slide.pptx
│       └── 4.2 Executive Summary Best Practices.pptx
├── case-study/                         # BCG case-study reference briefs
│   └── BCG_Task_1.pdf … BCG_Task_6.pdf
├── coc/
│   └── BCG_Data_Science_Certificate.pdf
├── assets/                             # BCG X logo + exported figures
├── LICENSE                             # MIT license
└── README.md                           # project documentation
```

---

## Task 1 — Business Understanding & Hypothesis Framing

Deliverable: `Solution/Task_1_email.pdf`

The engagement starts on the business side, not the code. Based on the
problem-context document, the engagement is framed as:

> **Hypothesis —** customer churn is sensitive to pricing. A change in the
> price of electricity (power and/or energy) may cause customers to churn.

The client-facing email:

1. **Restates the business problem** in the client's own terms and sets
   measurable context (churn rate, commercial impact).
2. **States the hypothesis** to be tested and explains why it matters for the
   retention strategy.
3. **Proposes the analytical plan** — validate price sensitivity through EDA,
   then build a predictive model to rank customers by churn risk.
4. **Anticipates the data needed:** customer contract/consumption data and
   monthly product-price history, both of which the client supplies for
   Task 2.

## Task 2 — Exploratory Data Analysis

Deliverables: `Solution/BCG_TASK2.ipynb`, `BCG_TASK2.pdf`
and `clean_data_after_eda.csv`

### Data sources

| Dataset    | Rows    | Columns | Contents                                        |
|------------|---------|---------|-------------------------------------------------|
| Client data| 14,606  | 26      | One row per client: consumption, contract dates, forecasts, margins, power, channel, campaign, `churn` |
| Price data | 193,002 | 8       | Monthly electricity price per client for 2015 (off-peak / peak / mid-peak, energy & power) |

Key columns in the client data: `cons_12m`, `cons_gas_12m`, `cons_last_month`,
`forecast_cons_12m`, `forecast_discount_energy`, `forecast_meter_rent_12m`,
`margin_gross_pow_ele`, `net_margin`, `nb_prod_act`, `num_years_antig`,
`pow_max`, `origin_up`, `channel_sales`, `has_gas` and the target **`churn`**
(churned within the next 3 months). Client IDs are hashed for privacy but keep
their commercial meaning.

### How the EDA works

1. **Cleaning.** Column names and string values are stripped, date columns are
   converted to `datetime`, and the `has_gas` flag is normalised.
2. **Univariate view.** Every feature group (consumption, forecast, margin,
   contract, price, power, campaign, products) is plotted as histograms and
   boxplots **split by churn**, exposing distribution shape, skewness and
   outliers.
3. **Feature engineering for the test.** To test the price hypothesis, the
   price history is rolled up into 12-month, 6-month and 3-month means per
   client (e.g. `mean_3m_price_off_peak_var`).
4. **Correlation check.** A heatmap of prices vs `churn` is produced and the
   aggregated price features are merged back into the client table.

### Key findings

- Consumption, margins and power are highly positively skewed, with many
  outliers and several 0-filled gas rows — handled later in feature
  engineering.
- Most clients have ~12 months of price history (Jan–Dec 2015).
- **Churn breaks the hypothesis:** churn shows almost no correlation with any
  price variable. In contrast, consumption, contract duration and the number
  of active products separate churners from non-churners.
- The cleaned, price-enhanced dataset is exported as
  `clean_data_after_eda.csv` (53 columns) for Task 3.

## Task 3 — Feature Engineering & Modeling

Deliverables: `Solution/feature_engineering.ipynb` and
`data/predicted_data.csv`

### How feature engineering works

- **Dec-Jan price deltas.** The difference between December and the preceding
  January for off-peak, peak and mid-peak prices is rebuilt from the
  colleague's snippet into `offpeak_diff_dec_january_energy` / `_power`,
  then extended to peak and mid-peak periods.
- **Period spread features.** Mean seasonal differences between pricing
  periods (`diff_var_off_peak`, `diff_var_off_mid`, `diff_var_mid_peak`, …
  and their `fix` counterparts) capture how much a tariff moves across the
  year.
- **Max prices.** `max_price_*` columns flag the worst (highest) price each
  client saw in 2015 — customers dislike increases even when averages look
  stable.
- **Time features.** `total_days` (contract length) plus each contract date
  measured in days from a reference date (`2016-12-01`); the original date
  columns are dropped.
- **Encoding.** `channel_sales`, `origin_up` and `has_gas` are one-hot encoded
  (one level each removed to avoid collinearity).
- **Skewness.** Columns with a standard deviation ≥ 50 are `log10(x + 1)`
  transformed, bringing heavy-tailed distributions close to normal.

### How the model works

1. **Split & scaling.** `X` (all features except `id`/`churn`) and `Y`
   (`churn`) are split 75:25 (`random_state = 42`); numeric columns are
   standardised with `StandardScaler`.
2. **Baseline.** A `RandomForestClassifier` (1,000 trees, `entropy` criterion)
   is trained and behaves like the client expects: ~90% accuracy.
3. **Tuning.** `GridSearchCV` is run over `n_estimators`, `max_depth` and
   `criterion`, then a second pass over `min_samples_split` and
   `min_samples_leaf`. Tuning barely moves the score, so the original
   parameters are kept.
4. **Evaluation.** Accuracy, confusion matrix, classification report, ROC/AUC,
   precision, recall, F1 and 5-fold cross-validation.

### Model evaluation

| Metric    | Random Forest |
|-----------|---------------|
| Accuracy  | 0.9036        |
| Precision | 0.8400        |
| Recall    | 0.0477        |
| F1 score  | 0.0903        |
| AUC score | 0.5300        |
| K-fold (5) | 0.9080       |

Confusion matrix: **TN 3,937 · FP 5 · FN 417 · TP 23** (test set of 4,382).

> Reading the results: the model is excellent at predicting *retention* but
> almost blind to *churn* — it only catches ~5% of churners. This is the
> classic symptom of a ~9:1 class imbalance, and it matters because a churn
> campaign only has value if it reaches the customers who actually leave.

### Feature importance

Consumption (`cons_12m`, `cons_gas_12m`, `cons_last_month`), net power and
net margin dominate the importance ranking; forecast rents and contract
duration follow. **Price variables are scattered near the bottom — a second,
independent confirmation that the pricing hypothesis is weak.**

### Discount impact analysis (20%)

With per-customer `churn_probability` from `predict_proba`, the notebook
quantifies a retention offer:

- `forecast_revenue = forecast_cons_12m × forecast_price_energy_off_peak
  + forecast_meter_rent_12m`
- Churners are assumed to leave within the first quarter →
  ≈ **87.6%** average revenue loss (`0.8757`).
- Customers with `churn_probability ≥ cutoff` receive a **20% discount** and
  are assumed to stay, keeping 80% of revenue.
- Sweeping the cutoff from 0 to 1, the strategy recovers **~$3.2k**.
- Restricting the offer to high-value customers
  (`retained_revenue > mean`) produces a similar outcome — value-focusing the
  discount does not clearly pay off at this churn rate.

Per-client predictions are exported to `data/predicted_data.csv`.

## Task 4 — Findings & Recommendation

Deliverable: `Solutions/Executive Summary.pdf`

The final executive summary translates the analysis into a board-ready
recommendation:

- **Reframe the retention strategy.** Price is a minor churn driver; a
  pricing-led retention campaign would be mis-targeted. Churn correlates most
  strongly with usage/power profile and customer tenure.
- **Fix the model, then use it.** Address the class imbalance (resampling,
  class weights, threshold tuning) before trusting churn *scores* to drive a
  campaign; the raw model under-reports real churners.
- **Discounts have measurable value.** A 20% offer to the riskiest customers
  recovers ~$3.2k in the simulation — and selectively discounting high-value
  clients keeps most of that upside at lower cost.
- Backed by the slide template and best-practice guidance in
  `Task Resources/`.

---

## Reproducing the Work

The analysis is notebook-based and was developed in Google Colab. To rerun it
locally:

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate

# 2. Install the data-science stack
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Launch Jupyter
jupyter notebook
```

Open the notebooks in dependency order and point the `read_csv` paths at the
files in each task's `Solution/` / `Task Resources/` folders:

| Step | Notebook / file | Reads |
|------|-----------------|-------|
| 1 | `Task 2 …/Solution/BCG_TASK2.ipynb` | `4.1 Client Data.csv`, `4.2 Price Data.csv` |
| 2 | (Task 2 output) → `clean_data_after_eda.csv` | — |
| 3 | `Task 3 …/Solution/feature_engineering.ipynb` | `clean_data_after_eda.csv`, `4.2 Price Data.csv` |
| 4 | `Task 3 …/Solution/data/predicted_data.csv` | produced by step 3 |

---

## Technologies Used

- **Python 3.10+**
- **pandas / NumPy** — cleaning, feature engineering, aggregations
- **Matplotlib / Seaborn** — EDA visualisations and evaluation plots
- **Scikit-learn** — `RandomForestClassifier`, `GridSearchCV`,
  `StandardScaler`, evaluation & cross-validation
- **Joblib** — model persistence
- **Jupyter / Google Colab** — notebook development environment

## Limitations

- **Class imbalance.** With ~9:1 non-churners to churners, the model reports
  ~90% accuracy while recalling only ~5% of real churners; accuracy alone
  overstates usefulness.
- **The price hypothesis was rejected.** Any recommendation that leaned on
  price as the retention lever is unsupported by this data.
- **Random Forest as a black box.** Feature importances give a ranking, not an
  explanation of *why*, and they do not capture interaction effects directly.
- **Single snapshot, not temporal.** The model is trained on one point in
  time; there is no train-on-past / test-on-future validation, so drift and
  seasonality are not measured.
- **No uplift modelling.** The discount simulation assumes every accepted
  offer retains the customer; responses are treated as deterministic.
- **Notebook workflow.** The analysis lives in notebooks rather than a
  reproducible pipeline/package, so rerunning end-to-end requires manual steps.

## Possible Improvements

- Close the imbalance gap: SMOTE/undersampling, `class_weight="balanced"`,
  and decision-threshold tuning to raise recall at an acceptable precision.
- Benchmark stronger learners (XGBoost, LightGBM, logistic regression with
  regularisation) and pick on ROC-PR rather than accuracy.
- Move to a proper train/validation/test split by *time* to estimate real
  uplift under drift.
- Swap discount simulation for true **uplift modelling** (e.g. two-model
  approach) so the offer targets customers whose behaviour actually changes.
- Use SHAP to explain individual churn scores and build trust with the
  client's retention team.
- Package the pipeline (feature store → model registry → scoring service) and
  schedule re-scoring on new monthly data.

---

## License

MIT — see [LICENSE](LICENSE).