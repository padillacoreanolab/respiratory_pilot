import pandas as pd, numpy as np
df = pd.read_csv("qc_windows.csv")
ref_std = df["std"].median(); ref_p99 = df["p99"].median()
df["flat"] = df["std"] < 0.25 * ref_std               # signal loss (cf. mask_signal_loss)
df["artifact"] = df["maxabs"] > 4 * ref_p99           # large motion/contact artifact
df["weak"] = df["band_frac"] < 0.6                     # breathing rhythm not dominant
df["good"] = ~(df.flat | df.artifact | df.weak)
s = df.groupby("session").agg(windows=("good", "size"), pct_good=("good", "mean"), flat=("flat", "mean"),
                              artifact=("artifact", "mean"), weak=("weak", "mean"), band=("band_frac", "median"))
s[["pct_good", "flat", "artifact", "weak"]] *= 100
s["category"] = np.where(s.pct_good >= 85, "Good", np.where(s.pct_good >= 60, "Somewhat bad", "Bad"))
print(f"ref window std {ref_std:.0f}, ref p99 {ref_p99:.0f} -> flat < {0.25*ref_std:.0f}, artifact > {4*ref_p99:.0f}")
print(s.round(1).to_string())
s.round(1).to_csv("qc_sessions.csv"); df.to_csv("qc_windows_labeled.csv", index=False)
