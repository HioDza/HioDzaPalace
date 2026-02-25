# sanitycheck-cli

## What is this?
`sanitycheck-cli` is a zero-tuning CLI tool for quickly assessing the statistical cleanliness of numeric columns in CSV datasets.  
It provides a fast, high-level overview before deeper analysis or modeling.

## When should I use it?
- Before exploratory data analysis (EDA)
- Before training statistical or machine learning models
- When you want a quick sanity check without manual inspection

## What it does NOT do
- It does not clean or modify data
- It does not model relationships or correlations between features
- It does not replace proper EDA or data validation pipelines

## Quick start
Run the tool on a CSV file:

```bash
sanitycheck data.csv
```

## Example output
📌Column problem: [0]
- [-]: Abnormal similarity
- [-]: NaN/Inf
- [Student_ID, Gender, Parental_Education, Internet_Access, Extracurricular_Activities, Performance_Level, Pass_Fail]: Non-numeric (ignored)
- [-]: Inconsistent value type

📌Row problem: [0]
- [-]: NaN/Inf (numeric columns only)

📌Top anomalous rows:
- Row 4867: score=0.625
- Row 82: score=0.624
- Row 1364: score=0.622
- Row 1482: score=0.619
- Row 4913: score=0.611

🔨 Final validation:
- [average entropy]: 0.915
- [average var]: 0.832
- clarity score: 1.000

## Interpretation tips
- Higher clarity scores indicate cleaner numeric data

- Anomalous rows are ranked, not classified — use them for inspection

- Non-numeric columns are ignored by design

- This tool is best used as a fast pre-analysis step