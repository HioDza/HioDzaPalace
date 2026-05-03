from rich import print

def metrics():
    print("[bold cyan]📌 Metrics[/bold cyan]")
    print("\n- Invalid value ratio: The proportion of invalid values (NaN, inf) to total values in the data.")
    print("\n- Inconsistent type column: Feature columns with inconsistent data types, such as rows 1-10 being floats but row 11 being strings. Inconsistent data, such as strings of letters, are converted to NA and then the ratio of their number to the total number of values in the column is calculated.")
    print("\n- Feature similarity: The similarity of two feature pairs through their correlation, optimized using the upper triangle operation.")
    print("\n- Row problem severity: A score of the severity of a row's problems based on the ratio of invalid values (finite values) and outliers (using the robust z-score with MAD).")
    print("\n- Entropy: The irregularity of a feature. The mean calculation results are presented normalized using 'H / max(H_max, ε)', and with H_max 'log2(len(hist))' or 1.0 if len(hist) < 1. Other forms of presentation are the top 5 highest or the mean entropy of all features.")
    print("\n- Variance (spread): How spread out the data values are. The mean calculation results are presented normalized using 'var / (var + baseline + ε)', and with baseline '(IQR / σ)^2' (σ = 1.349) or STD^2 features if IQR < 0. Other forms of presentation are the top 5 highest or the mean var of all features.")
    print("\n- Multicollinearity: Multicollinearity is calculated using the VIF (Variance Inflation Factor). The calculated mean is presented normalized using '1 - tanh(VIF / 10). Another form of presentation is the top 5 highest of all features.")
    print("\n- Sparsity: The proportion of zero-value (0) data points to the total data set. This is calculated by taking into account the data size as a penalty of '0.7 * sparsity_ratio + 0.3 * dim_penalty'. The penalty is calculated as 'n_feature / (n_feature + n_sample)'.")
    print("\n- Class override ratio: The proportion of data points with the exact same feature but different classes in the dataset.")
    print("\n- Class imbalance: The degree of imbalance in the classes, calculated using the normalized 'Gini' criterion formula 'gini / (1 - 1 / n_class)'.")
    print("\n- Shapiro-Wilk and Kolmogorov-Smirnov scores: Results of tests of normality of distribution from raw p-values.")
    print("\n- Structure normality: Based on kurtosis and skewness using '1 - (0.5 * skew_score + 0.5 * kurt_score)' with skew_score = np.tanh(skew_mean / 2) and kurt_score = np.tanh(kurt_mean / 5). The mean is taken from the kurtosis or skewness scores of all features.")
    print("\n- Cleanliness score: Taken from the weighted accumulation of the invalid value ratio (0.30), inconsistent column type (0.20), abnormal column similarity (0.20), and row problem severity (0.30).")
    print("\n- Linear signal: Calculated using MI (Mutual Information) and ANOVA (Analysis of Variance) via the formula 'norm_f_score / (mean_mi + ε)' with norm_f_score = tanh(mean(f_scores) / 10).")