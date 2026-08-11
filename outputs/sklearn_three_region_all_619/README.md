# All-eligible-bout scikit-learn analysis

This is the primary all-bout analysis. It uses all 619 eligible `exp1`/`exp5`
bouts from the seven histology-confirmed subjects, restricted to power and
coherence from MD, mPFC, and NAc. No bouts were downsampled.

## Input audit

- Unique bouts: 619
- Neural features per bout: 30
- Long-format feature rows: 18,570
- Duplicate feature keys: 0
- Events: `exp1 sniff` (partner A) and `exp5 sniff` (partner B) only
- Subjects: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, and 4.1

## Nested subject validation

Each outer fold uses five complete subjects for model-selection training, one
complete subject for L2 validation, and one untouched subject for testing.
Every subject serves as the test subject once. The repository does not balance
or downsample bout counts during fitting.

This prevents cross-subject leakage, but it does not give every training subject
equal weight: a subject with more bouts contributes more rows to the logistic
loss. Equal subject weighting would be a separate modeling choice and would no
longer be an exact replication of the repository.

## Results

| Task | Mean accuracy | Mean balanced accuracy | Mean AUC | AUC SEM |
|---|---:|---:|---:|---:|
| Cagemate identity A/B | 0.635 | 0.625 | 0.740 | 0.088 |
| Novel identity A/B | 0.608 | 0.564 | 0.661 | 0.125 |
| Combined identity A/B | 0.583 | 0.544 | 0.581 | 0.098 |
| Familiarity: cagemate/novel | 0.623 | 0.629 | 0.673 | 0.093 |

These are descriptive means and SEM across seven held-out subjects. They are not
formal p-values. The identity results show that identity-related neural signal
is present in the complete dataset and must be considered when interpreting the
familiarity result.

The earlier 140-bout analysis is retained only as a balanced/downsampled
sensitivity analysis; it should not be treated as the primary input.
