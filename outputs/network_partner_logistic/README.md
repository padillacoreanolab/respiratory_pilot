# Power/coherence partner classification

Outcome coding is `cagemate = 0` and `novel = 1`. The analysis pivots the long
table to one row per interaction, producing 30 neural predictors: power and
coherence for five frequency bands across three regions or region pairs.

The primary model is an L2-regularized logistic regression. Neural predictors
are standardized within each training fold, partner identity (`A`/`B`) is
included as a nuisance covariate, and performance is evaluated with
leave-one-subject-out cross-validation. Thus, no trials from a held-out subject
are used to fit its model or preprocessing. Network-only and identity-only
models are included as benchmarks.

`full_model_coefficients.csv` contains coefficients from a fit to all data.
They are descriptive feature weights, not p-values or independent hypothesis
tests. `loso_predictions.csv` contains only out-of-sample predictions.

Run from the repository root with:

```powershell
python scripts\network_partner_logistic.py `
  "C:\path\to\safe_balanced_3_regions.csv" `
  --output-dir outputs\network_partner_logistic
```
