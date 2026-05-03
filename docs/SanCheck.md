# SanCheck (Sanity Check)

`Sanity Check` is a program that helps provide a statistical overview of data, including several things such as multicollinearity via VIF (Variance Inflation Factor), invalid value ratio, etc. The SanCheck project was created to address the inefficiency of repetitive preprocessing tasks in machine learning. The main objective of SanCheck is to provide a statistical overview of the data as it is without cleaning or outputting transformed input data.

## Metrics
The metrics checked by SanCheck are as follows:
- **Invalid Value Ratio**: The number of invalid values ​​(NaN, inf) compared to the total values ​​in the data.

- **Inconsistent Type Column**: Feature columns with inconsistent data types, such as rows 1-10 being floats but row 11 being strings. Inconsistent data, such as strings of letters, are converted to NA and then the ratio of their number to the total number of values ​​in the column is calculated.

- **Feature Similarity**: The similarity of two feature pairs through their correlation, optimized using the upper triangle operation.

- **Row Problem Severity**: A score of the severity of a row's problems based on the ratio of invalid values ​​(finite values) and outliers (using the robust z-score with MAD).

- **Entropy**: The irregularity of a feature. The mean calculation results are presented normalized using 'H / max(H_max, ε)', and with H_max 'log2(len(hist))' or 1.0 if len(hist) < 1. Other forms of presentation are the top 5 highest or the mean entropy of all features.

- **Variance (spread)**: How spread out the data values ​​are. The mean calculation results are presented normalized using 'var / (var + baseline + ε)', and with baseline '(IQR / σ)^2' (σ = 1.349) or STD^2 features if IQR < 0. Other forms of presentation are the top 5 highest or the mean var of all features.

- **Multicollinearity**: Multicollinearity is calculated using the VIF (Variance Inflation Factor). The calculated mean is presented normalized using '1 - tanh(VIF / 10). Another form of presentation is the top 5 highest of all features.

- **Sparsity**: The proportion of zero-value (0) data points to the total data set. This is calculated by taking into account the data size as a penalty of '0.7 * sparsity_ratio + 0.3 * dim_penalty'. The penalty is calculated as 'n_feature / (n_feature + n_sample)'.

- **Class Override Ratio**: The proportion of data points with the exact same feature but different classes in the dataset.

- **Class Imbalance**: The degree of imbalance in the classes, calculated using the normalized 'Gini' criterion formula 'gini / (1 - 1 / n_class)'.

- **Shapiro-Wilk and Kolmogorov-Smirnov scores**: Results of tests of normality of distribution from raw p-values.

- **Structure normality**: Based on kurtosis and skewness using '1 - (0.5 * skew_score + 0.5 * kurt_score)' with skew_score = np.tanh(skew_mean / 2) and kurt_score = np.tanh(kurt_mean / 5). The mean is taken from the kurtosis or skewness scores of all features.

- **Cleanliness score**: Taken from the weighted accumulation of the invalid value ratio (0.30), inconsistent column type (0.20), abnormal column similarity (0.20), and row problem severity (0.30).

- **Linear signal**: Calculated using MI (Mutual Information) and ANOVA (Analysis of Variance) via the formula 'norm_f_score / (mean_mi + ε)' with norm_f_score = tanh(mean(f_scores) / 10).

## Arguments

- **csv**: CSV file path to be processed.

- **target**: Target column name (classification only).

- **--plot-chunk**: Number of columns per chunk for plotting, or 'all'.

- **--download-plot**: Download plots as PNG files.

- **--no-plot**: Skip plotting (overrides --download-plot).

- **--metrics-info**: Show detailed explanation of metrics.

- **--get-json**: Output the report as JSON instead of printing to console.

- **--mute**: Mute safeguards and automatically fallback to the safest option.