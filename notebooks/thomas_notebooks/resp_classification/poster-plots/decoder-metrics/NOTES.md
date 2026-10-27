# Decoder metrics — methodology notes

**What is being decoded.** Two binary classifications from BreathMetrics respiration features:
- **Valence**: Positive (RI1) vs Negative (RI2) social interaction.
- **Rank**: Dominant vs Subordinate cagemate (Aim1 baseline cagemate cohort, 7 mice / 4 cages).

Each is run on two window types: **session** (3 s windows sampled across the recording) and
**interaction** (sniff-bout windows, both actors).

**Train / validation / evaluation.** There is no separate hold-out test split. Performance is
estimated by **leave-one-group-out cross-validation**, where the group is the biological unit:
- Valence → **leave-one-mouse-out (LOSO)**: valence is a within-mouse contrast (same mouse does
  RI1 and RI2), so an entire mouse is held out and predicted from models trained on the other mice.
- Rank → **leave-one-cage-out (LOCO)**: both cagemates of a pair are held out together, so no
  mouse from the test cage (in either social role) is seen during training.

In every fold the held-out group is the **validation set** and its windows are predicted
out-of-fold; concatenating all folds gives one out-of-fold prediction per window. Model
hyper-parameters are **fixed** (not tuned), so there is no inner tuning loop and no optimism from
peeking at the test groups. `class_weight="balanced"` corrects class imbalance inside training.

**Models & kernels.** Three fixed (untuned) estimators, all `class_weight="balanced"`:
- **LogisticRegression** — LINEAR (L2, liblinear).
- **SVM** — **NON-LINEAR: RBF (Gaussian) kernel** (`kernel="rbf"`, C=1, gamma="scale").
- **RandomForest** — NON-LINEAR (300 trees, depth 5). So two of the three models are non-linear;
  a non-linear kernel (RBF-SVM) WAS used and is reported alongside the linear LogReg and RF.

**Metrics.** Balanced accuracy (mean of per-class recall) and macro F1, computed per fold then
averaged (mean ± std across folds); chance = 0.5. The bar panels show per-fold **mean ± SEM**
(std/√folds); the CSV keeps both std and SEM. Features are respiration-only — BORIS behavior
columns (n_bouts_*/total_dur_*_sniffing) are excluded so behavior can't leak the label.

**Caveats.**
- Small cohorts. Rank LOCO has only 4 folds; **cage 3 = singleton {3_5}** (Subordinate only) makes
  one fold single-class and degenerate, inflating the std.
- Valence **interaction** windows are sparse for the negative class, so that variant is noisy;
  valence **session** is the reliable valence estimate.
- Feature panels: signed standardized **LINEAR L2-LogReg** coefficients fit on the full best variant
  (bars show |coefficient|); positive-coef features define one class, negative the other. These are
  from the LINEAR model even when the best-performing bar is the non-linear SVM/RF, and are
  **multivariate partial effects**, NOT group means.
- **Do NOT read direction off the RANK feature panel.** (a) The respiration features are highly
  collinear (breathing rate, breath count, IBI all measure frequency), so LogReg splits the weight
  and coefficient SIGNS flip — e.g. "Breathing Rate" can land on the Dominant side while "Breath
  Count" (the same thing) lands on Subordinate. (b) The rank model is at CHANCE, so its coefficients
  fit noise. Trust the univariate whole-session resp-rate plot (Dominants breathe slower) for
  direction, not these coefficients. Valence (~0.81) coefficients are meaningful; rank's are not.
