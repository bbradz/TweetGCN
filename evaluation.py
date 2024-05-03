from scipy.stats import ttest_rel

scores_A = [0.90, 0.85, 0.88, 0.90, 0.87]
scores_B = [0.85, 0.80, 0.83, 0.82, 0.84]

scores_A = .90
scores_B = .85

t_stat, p_value = ttest_rel(scores_A, scores_B)
print(f"T-statistic: {t_stat}, P-value: {p_value}")