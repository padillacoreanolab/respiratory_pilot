"""Numerical summary of the real-data comparison (documented drops excluded).
  python summarize.py <out_dir>   -> <out_dir>/summary.md and summary_*.csv"""
import os
import sys

import numpy as np
import pandas as pd

from adaptive_breath_detection import FEATURE_STATUS

OUT = sys.argv[1]
EXCLUDE = {"RI1_5_13", "RI2_5_13", "BLRI_5_13"}
lines = []


def say(s=""):
    print(s)
    lines.append(s)


det = pd.read_csv(os.path.join(OUT, "detection_per_recording.csv"))
det = det[~det.key.isin(EXCLUDE)]
say("# Legacy vs adaptive_prominence on real recordings\n")
say("Rank: evaluation minutes only (even minutes; odd minutes were used for parameter selection).")
say("Valence: all minutes of every usable recording (never used for tuning). Excluded: 5_13 "
    "(documented drop; BLRI_5_13 has no resp dataset).\n")
say("## Breath detection (valid evaluation samples)\n")
for coh, g in det.groupby("cohort"):
    w = lambda c, wt: np.average(g[c], weights=g[wt])  # noqa: E731
    say(f"**{coh}** ({len(g)} recordings, {g.eval_live_s.sum() / 60:.0f} valid minutes)")
    say(f"- breaths: legacy {g.n_legacy.sum():,}, adaptive {g.n_adaptive.sum():,} "
        f"({100 * (g.n_adaptive.sum() / g.n_legacy.sum() - 1):+.1f}%)")
    say(f"- found only by legacy: {g.legacy_only.sum():,}; only by adaptive: {g.adaptive_only.sum():,}")
    say(f"- interval ~2x neighbours (missed-breath signature): legacy {100 * w('doubling_legacy', 'n_legacy'):.1f}%, "
        f"adaptive {100 * w('doubling_adaptive', 'n_adaptive'):.1f}%")
    say(f"- interval <0.6x neighbours (split/false-peak signature): legacy {100 * w('halving_legacy', 'n_legacy'):.1f}%, "
        f"adaptive {100 * w('halving_adaptive', 'n_adaptive'):.1f}%")
    say(f"- NeuroKit2 (khodadad2018) peaks also found: by legacy {100 * w('nk_matched_by_legacy', 'n_neurokit'):.1f}%, "
        f"by adaptive {100 * w('nk_matched_by_adaptive', 'n_neurokit'):.1f}%")
    say(f"- detector peaks also found by NeuroKit2: legacy {100 * w('legacy_matched_by_nk', 'n_legacy'):.1f}%, "
        f"adaptive {100 * w('adaptive_matched_by_nk', 'n_adaptive'):.1f}%")
    say(f"- samples marked invalid: {100 * np.average(g.invalid_frac, weights=g.eval_live_s):.2f}% (time-weighted)\n")
per = det[["key", "cohort", "subject", "label", "legacy_rate_hz", "adaptive_rate_hz", "legacy_only", "adaptive_only",
           "invalid_frac"]].copy()
per["rate_change_pct"] = 100 * (per.adaptive_rate_hz / per.legacy_rate_hz - 1)
per.to_csv(os.path.join(OUT, "summary_per_recording.csv"), index=False)
say("### Per recording (rate = breaths per valid evaluation second)\n")
say(per.round(3).to_markdown(index=False))
say("\n### Valence by condition (mean over recordings)\n")
say(per[per.cohort == "valence"].groupby("label")[["legacy_rate_hz", "adaptive_rate_hz", "rate_change_pct"]].mean()
    .round(2).to_markdown())

amp = pd.read_csv(os.path.join(OUT, "detection_by_amplitude.csv"))
amp = amp[~amp.key.isin(EXCLUDE)]
amp["cohort"] = np.where(amp.key.str.startswith("CM_"), "rank", "valence")
say("\n## By local amplitude (local 1 s range at the peak / recording median)\n")
say(amp.groupby(["cohort", "stratum"])[["legacy", "adaptive", "legacy_only", "adaptive_only"]].sum().to_markdown())

acf = pd.read_csv(os.path.join(OUT, "acf_windows.csv"))
acf = acf[~acf.key.isin(EXCLUDE) & acf.eval_window & (acf.qc_invalid_frac == 0) & acf.acf_rate.notna()]
say("\n## Independent rate check (autocorrelation; no peak picking)\n")
for c, nm in [("rate_legacy", "legacy"), ("rate_adaptive", "adaptive")]:
    e = (acf[c] - acf.acf_rate) / acf.acf_rate
    say(f"- {nm}: median error {100 * e.median():+.1f}%, within 15% in {100 * (e.abs() < .15).mean():.0f}% "
        f"of {e.notna().sum()} windows")
say("  (the ACF reports the dominant period, which favours the larger of alternating breaths, so it is a"
    " conservative check for a detector that adds small breaths)")

fw = pd.read_csv(os.path.join(OUT, "features_windows.csv"))
fw = fw[~fw.key.isin(EXCLUDE) & fw.eval_window]
fw["cohort"] = np.where(fw.key.str.startswith("CM_"), "rank", "valence")
rows = []
feats = [c[:-len("__legacy")] for c in fw.columns if c.endswith("__legacy")]
for f in feats:
    a, b = fw[f + "__legacy"], fw[f + "__adaptive"]
    ok = a.notna() & b.notna()
    rel = ((b - a) / a.abs().replace(0, np.nan))[ok]
    rows.append(dict(feature=f, status=FEATURE_STATUS.get(f, "?"), n=int(ok.sum()),
                     spearman=a[ok].corr(b[ok], method="spearman"), median_change_pct=100 * rel.median(),
                     median_abs_change_pct=100 * rel.abs().median(),
                     legacy_nan_pct=100 * a.isna().mean(), adaptive_nan_pct=100 * b.isna().mean()))
ft = pd.DataFrame(rows)
ft.to_csv(os.path.join(OUT, "summary_feature_changes.csv"), index=False)
say("\n## Downstream window features, same windows (rank: your saved 3 s windows in evaluation minutes; "
    "valence: 100 windows/session from sample_random_nonoverlapping_windows, seed 42)\n")
say(ft.round(3).to_markdown(index=False))
for coh, g in fw.groupby("cohort"):
    say(f"\n{coh}: windows passing the >=5 breath gate: legacy {(g.n_breaths__legacy >= 5).sum()}/{len(g)}, "
        f"adaptive {(g.n_breaths__adaptive >= 5).sum()}/{len(g)}; windows with any invalid sample "
        f"{(g.qc_invalid_frac > 0).sum()}")

q = pd.read_csv(os.path.join(OUT, "session_quality.csv"))
say("\n## Whole-session sanity check (absolute units, compare across sessions)\n")
say(q.sort_values("median_cycle_amplitude").head(5).round(2).to_markdown(index=False))
pol = pd.read_csv(os.path.join(OUT, "polarity.csv"))
pol = pol[~pol.key.isin(EXCLUDE)]
say("\n## Polarity evidence (weak)\n")
say(f"- median positive-lobe duration {1000 * pol.median_positive_lobe_s.median():.0f} ms vs negative-lobe "
    f"{1000 * pol.median_negative_lobe_s.median():.0f} ms (negative longer in "
    f"{(pol.median_negative_lobe_s > pol.median_positive_lobe_s).mean() * 100:.0f}% of recordings)")
say(f"- pauses after the negative lobe in {100 * pol.frac_pause_after_negative_lobe.median():.1f}% of breaths vs after "
    f"the positive lobe {100 * pol.frac_pause_after_positive_lobe.median():.1f}% (median over recordings)")
say("- Both fit positive = inspiration (mouse expiration is usually longer and pauses are mostly "
    "end-expiratory). The sensor's physical polarity is NOT recorded in the H5 metadata.")
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
