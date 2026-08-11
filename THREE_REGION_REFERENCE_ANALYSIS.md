# Replicating the three-region logistic-regression workflow

> **Updated input decision:** The `safe_balanced_3_regions.csv` file contains a
> 140-bout downsample. The primary all-eligible analysis now uses
> `Data/all_eligible_3_regions_7_mice.csv` (619 bouts). Results from the 140-bout
> file should be treated only as a downsampled sensitivity analysis. See
> `outputs/sklearn_three_region_all_619/README.md`.

This project has been run against `safe_balanced_3_regions.csv` using the
workflow from:

https://github.com/junwoozy/three-region-logistic-regression

## Study design

- `exp1 sniff` is partner identity A.
- `exp5 sniff` is partner identity B.
- Events 2, 3, and 4 are not present or analyzed.
- Neural predictors are the 30 power/coherence features from five frequency
  bands, three power regions, and three coherence pairs.
- Identity-control models predict A versus B separately for cagemate and novel
  interactions.
- The primary familiarity model retains both A and B and predicts
  `cagemate = 0` versus `novel = 1`.
- Validation is nested by subject: five training subjects, one validation
  subject, and one untouched test subject. Every subject serves as test once.

Partner A/B is **not** included as a predictor in the familiarity model. Its
effect is evaluated through the separate identity-control models.

## Reproduce the complete run

Clone the reference repository and place the CSV in its `Data` directory:

```powershell
git clone https://github.com/junwoozy/three-region-logistic-regression.git
Set-Location three-region-logistic-regression
New-Item -ItemType Directory -Force Data
Copy-Item "C:\Users\sjs93\Downloads\safe_balanced_3_regions.csv" `
  "Data\safe_balanced_3_regions.csv"
python -m pip install -r requirements.txt
python logistic_identity_ab_model_start.py `
  --split-strategy nested_subject_loso `
  --seed 42
```

The complete run includes the cagemate identity, novel identity, combined
identity, and familiarity models, plus all region and frequency-band ablations.

## Primary outputs in this project

Results are under:

`Logistic_Regression_Results/validation_auc_selected_5_train_1_validation_1_test/nested_subject_loso/`

Start with:

- `familiarity/csv/familiarity_summary.csv`: primary familiarity performance.
- `comparison/csv/familiarity_vs_average_identity_auc_summary.csv`: familiarity
  versus the A/B identity control.
- `familiarity/csv/familiarity_top_coefficients.csv`: descriptive feature weights.
- `familiarity/ablation/region/csv/region_factor_ranking.csv`: region importance.
- `familiarity/ablation/band/csv/band_factor_ranking.csv`: frequency-band importance.

The older `outputs/network_partner_logistic/` folder came from a simpler model
that does not exactly reproduce the reference repository and should not be used
for the replication.
