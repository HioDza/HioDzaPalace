[![PyPI version](https://badge.fury.io/py/sancheck.svg)](https://pypi.org/project/sancheck/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sancheck/blob/main/LICENSE)

# sancheck

`sancheck` is a minimal-tuning CLI tool for quickly assessing the statistical overview of CSV datasets.  
It provides a fast, high-level overview before deeper analysis or modeling.

## When should I use it?
- Before exploratory data analysis (EDA)
- Before training statistical or machine learning models
- When you want a quick sanity check without manual inspection

## What it does NOT do
- It does not clean or modify data
- It does not model relationships
- It does not replace proper EDA or data validation pipelines

## Quick start
Run the tool on a CSV file:

```bash
sancheck [csv_path] [target_col_name]
```

## Example output with assets/data_example.csv

### 💬 Terminal output

```bash
📊 Dataset Summary
- Valid numeric columns: 5
- Ignored non-numeric columns: 2
- Inferred task type: classification

📌 Column problems
- Column with NaN/Inf/invalid: 0
  - math_score: invalid=0/10 (0.000)
  - science_score: invalid=0/10 (0.000)
  - english_score: invalid=0/10 (0.000)
  - total_score: invalid=0/10 (0.000)
  - age: invalid=0/10 (0.000)
- Type inconsistency column: 0
  - math_score: bad_type=0 (0.000)
  - science_score: bad_type=0 (0.000)
  - english_score: bad_type=0 (0.000)
  - total_score: bad_type=0 (0.000)
  - age: bad_type=0 (0.000)
- Similar feature pairs (|corr| >= 0.95):
  - Severity similarity: 0.996
  - math_score <-> total_score: |corr|=0.998
  - english_score <-> total_score: |corr|=0.993
  - science_score <-> total_score: |corr|=0.992
  - math_score <-> english_score: |corr|=0.990
  - math_score <-> science_score: |corr|=0.988
  - science_score <-> english_score: |corr|=0.972
  - Affected columns: english_score, math_score, science_score, total_score

📌 Row problems
- Problematic rows (NaN/Inf): 0/10
- Severity row: 0.141
  - row 2: score=0.667, invalid=False
  - row 3: score=0.594, invalid=False
  - row 5: score=0.580, invalid=False
  - row 6: score=0.576, invalid=False
  - row 7: score=0.555, invalid=False

📌 Distribution and interpretation
- High entropy means the distribution is more even/complex; it's not automatically 'noise', it can also be multimodal.
- High spread score means the data is more dispersed robustly compared to its central tendency.

  Top entropy (normalized 0-1):
  - math_score: entropy=0.960 (very spread / more uniform or complex distribution)
  - science_score: entropy=0.960 (very spread / more uniform or complex distribution)
  - total_score: entropy=0.960 (very spread / more uniform or complex distribution)
  - age: entropy=0.960 (very spread / more uniform or complex distribution)
  - english_score: entropy=0.859 (very spread / more uniform or complex distribution)

  Top spread (normalized 0-1):
  - science_score: spread_score=0.562 (wide / large variation), var=0.006, iqr=0.092
  - total_score: spread_score=0.538 (wide / large variation), var=0.057, iqr=0.298
  - math_score: spread_score=0.528 (wide / large variation), var=0.007, iqr=0.108
  - english_score: spread_score=0.508 (wide / large variation), var=0.006, iqr=0.103
  - age: spread_score=0.503 (wide / large variation), var=2.222, iqr=2.000

📌 Relation and structure
- VIF mean (normalized): 1.000 (very high)
- VIF per-feature
  - math_score: VIF=inf
  - science_score: VIF=inf
  - english_score: VIF=inf
  - total_score: VIF=inf
  - age: VIF=inf
- linear signal score: 1.000 (very high)
- sparsity: 0.100 (low)
- class imbalance ratio: 0.040 (low)
- class override ratio: 0.000 (low)

📌 Normality
- Shapiro-wilk and KS test score per-feature:
  - math_score: Shapiro=0.902 | KS=0.997
  - science_score: Shapiro=0.903 | KS=0.970
  - english_score: Shapiro=0.746 | KS=0.987
  - total_score: Shapiro=0.912 | KS=0.975
  - age: Shapiro=0.341 | KS=0.925
- Structure normality (based on skewness and kurtosis): 0.838 (very high)

🧼 Cleanineess status
- cleanliness score: 0.759 / 1.000
- cleanliness label: fairly clean
- missing severity: 0.000
- type severity: 0.000
- similarity severity: 0.996
- row severity: 0.141

📊 Dataset-level distribution summary
- avg entropy: 0.940
- avg spread score: 0.528

⏱️ Elapsed time: 10.01 seconds

❌ Exception(s) during process: []

⚠️ Warning(s) during process: [{'type': 'RuntimeWarning', 'message': 'divide by zero encountered in scalar divide', 'where': 'compute_vif computation'}, 
{'type': 'RuntimeWarning', 'message': 'divide by zero encountered in scalar divide', 'where': 'compute_vif computation'}, {'type': 'RuntimeWarning', 'message':
'divide by zero encountered in scalar divide', 'where': 'compute_vif computation'}, {'type': 'RuntimeWarning', 'message': 'divide by zero encountered in scalar
divide', 'where': 'compute_vif computation'}]
```

### 📊 Boxplot visualization
![Boxplot Example](assets/boxplot_numeric_columns(all_features).png)

### 🔥 Heatmap visualization
![Heatmap Example](assets/heatmap_correlation(all_features).png)


## Interpretation tips
- Higher clarity scores indicate cleaner numeric data

- Anomalous rows are ranked, not classified — use them for inspection

- Non-numeric columns are ignored by design

- This tool is best used as a fast pre-analysis step

- Some metric may produce infinity values for some reason, such as VIF metric if pair of feature has a very high correlation score