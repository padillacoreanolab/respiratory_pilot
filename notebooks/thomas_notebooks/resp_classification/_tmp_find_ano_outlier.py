import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from resp_helper_functions import build_interaction_boris_catalog, load_clean_boris

aim1_h5_dir = Path(
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs"
)
aim1_boris_dir = Path(
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_boris\baseline csv"
)
bla_cm_h5_dir = Path(
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\BLA_ChR_resp_pilot1 (SS)\BLA_resp_ChR_cm_hc\h5_outputs"
)
bla_cm_boris_dir = Path(
    r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\BLA_ChR_resp_pilot1 (SS)\BLA_resp_ChR_cm_boris"
)

_, boris_paths, _, _ = build_interaction_boris_catalog(
    aim1_h5_dir,
    aim1_boris_dir,
    bla_cm_h5_dir,
    bla_cm_boris_dir,
    aim1_skip_boris_stems={"CM_s3_6_sub3_5_20250623_174348"},
)

behavior_files = pd.DataFrame()
for trial_key, boris_path in boris_paths.items():
    cohort = "Aim1" if trial_key.startswith("AIM1") else "BLA"
    cleaned = load_clean_boris(boris_path, subject_only=False)
    if cleaned.empty:
        continue
    cleaned["source_file"] = trial_key
    cleaned["cohort"] = cohort
    cleaned["boris_path"] = str(boris_path)
    behavior_files = pd.concat([behavior_files, cleaned], ignore_index=True)

sniff_kw = ["sniff", "sniffing"]
is_sniff = behavior_files["Behavior"].str.lower().str.contains("|".join(sniff_kw))
sniffing = behavior_files[is_sniff].copy()
low_files = sniffing.groupby("source_file").size().pipe(lambda s: s[s < 10].index.tolist())
plot_df = sniffing[~sniffing["source_file"].isin(low_files)].copy()

ano = plot_df[plot_df["Behavior"] == "anogenital sniffing"].copy()
print("Anogenital bouts total:", len(ano))
print("\nDuration describe (all cohorts):")
print(ano["Duration"].describe())

print("\n=== Top 20 longest anogenital bouts ===")
cols = ["cohort", "source_file", "Initiator", "Start", "Stop", "Duration", "boris_path"]
top = ano.sort_values("Duration", ascending=False).head(20)[cols]
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 200)
print(top.to_string(index=False))

print("\n=== Aim1 anogenital per session ===")
aim1_ano = ano[ano["cohort"] == "Aim1"]
sess = aim1_ano.groupby("source_file").agg(
    count=("Duration", "size"),
    total_dur=("Duration", "sum"),
    max_dur=("Duration", "max"),
)
print(sess.sort_values("total_dur", ascending=False).to_string())

print("\n=== Cohort comparison (anogenital bout duration) ===")
for cohort in ["Aim1", "BLA"]:
    sub = ano[ano["cohort"] == cohort]
    print(
        f"{cohort}: n={len(sub)}, "
        f"median={sub['Duration'].median():.3f}s, "
        f"mean={sub['Duration'].mean():.3f}s, "
        f"max={sub['Duration'].max():.3f}s"
    )

# IQR outlier detection within anogenital
q1, q3 = ano["Duration"].quantile([0.25, 0.75])
iqr = q3 - q1
thresh = q3 + 3 * iqr
outliers = ano[ano["Duration"] > thresh]
print(f"\n=== IQR outliers (>{thresh:.2f}s) ===")
print(outliers[cols].to_string(index=False))

# What fraction of Aim1 anogenital time is from longest bout per session?
print("\n=== Longest bout as % of session anogenital time (Aim1) ===")
for sf, g in aim1_ano.groupby("source_file"):
    mx = g.loc[g["Duration"].idxmax()]
    pct = 100 * mx["Duration"] / g["Duration"].sum()
    print(f"{sf}: longest={mx['Duration']:.1f}s ({pct:.0f}% of session ano time), start={mx['Start']:.1f}s")
