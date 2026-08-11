# Public-library replication

This analysis replaces the reference repository's custom NumPy logistic
optimizer with `sklearn.linear_model.LogisticRegression` (scikit-learn 1.5.2).
Everything else is held constant: the CSV, 30 power/coherence features,
outcome coding, seed 42 subject roles, five training subjects, one validation
subject, one test subject, training-only standardization, L2 candidates, and
validation selection rule.

The custom objective uses `lambda` in mean log loss plus
`0.5 * lambda * ||weights||^2`. For scikit-learn the matching conversion is
`C = 1 / (number_of_fit_rows * lambda)`.

## Primary familiarity result

- Custom implementation: mean test AUC 0.641; accuracy 0.629.
- Scikit-learn: mean test AUC 0.644; accuracy 0.629.
- Held-out label agreement: 100% (140/140).
- Probability correlation: 0.994.
- Mean absolute probability difference: 0.0075.
- Standardized coefficient correlation: 0.963.

This is a close numerical replication of the primary cagemate-versus-novel
result. Six of seven folds selected the identical L2 value. The remaining fold
selected 0.01 with scikit-learn versus 0.001 with the custom optimizer.

The identity-model probabilities agree less closely, especially in folds that
select very weak regularization. This is consistent with solver/convergence
sensitivity in a small-sample, 30-feature problem. It does not change the main
finding that the familiarity result replicates with a standard public library.

`sklearn_model_summary.csv` contains task-level results.
`custom_vs_sklearn_comparison.csv` contains direct prediction comparisons.
The task-specific fold, prediction, coefficient, and L2-selection CSVs provide
the full audit trail.
