"""Writes peak_detection_workbench.ipynb (run once; afterwards edit the notebook directly)."""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Peak detection workbench

Edit the detector in cells **A**, **B** and **C**, then **Run All**. Every plot uses your edited detector.

| Cohort | Recordings | Used in |
|---|---|---|
| Cohort 1 (Jun 2025): cagemate + valence (BLRI, RI1, RI2) | 7 + 23 | Sections 2, 3 |
| Cohort 3 (Jul 2026): cagemate + RI1 | 11 + 44 | Sections 1, 6 |

- Cohort 2 (Jan 2026 BLA pilot) isn't used. Its cells in Section 6 are commented out.
- Cohort 3's Dropbox folder is `2026/BLA_ChR_resp_cohort2`.
- Signals: raw 20 kHz H5, filtered to 0.1-20 Hz and downsampled to 400 Hz. Subject 5_13 is left out.
"""))

cells.append(md("""## Setup

Run with the **biopipeline-env** kernel. This cell finds the notebook's folder so the helper module imports.
"""))

cells.append(code('''import os
import sys
from pathlib import Path

_cands = [Path(globals().get("__vsc_ipynb_file__", "")).parent, Path.cwd(),
          Path.cwd() / "notebooks/thomas_notebooks/resp_classification/detector_validation"]
NB_DIR = next(p.resolve() for p in _cands if (p / "workbench_support.py").exists())
os.chdir(NB_DIR)
sys.path.insert(0, str(NB_DIR))
print("working in", NB_DIR)
'''))

cells.append(md("""## Cell A: signal validity

Decides which samples are usable. Detection only runs inside the valid segments, and nothing is linked across them.
"""))

cells.append(code('''# ============================================================================
# CELL A: signal validity (what counts as dead or bad signal)
#   validity_mask() decides which samples are usable. Detection only runs inside the
#   valid segments it returns, and nothing is linked across them.
# ============================================================================
import numpy as np
import pandas as pd

VALIDITY_PARAMS = dict(
    block_s=1.0,              # screening block length (s)
    block_hop_s=0.25,         # hop between blocks (s)
    flat_rel_range=0.03,      # block 5-95% range < this x session median range ...
    flat_acf_max=0.5,         # ... and autocorrelation peak (60-500 ms) below this  -> dead / dropout
    low_amp_rel_range=0.15,   # valid but flagged low-amplitude below this
    artifact_rel_range=8.0,   # block range > this x session median                 -> artifact
    artifact_pad_s=0.25,      # padding around artifacts (s)
    min_segment_s=1.0,        # valid runs shorter than this are dropped
)


def running_mean(x, n):
    """Centred running mean; edges average the samples that exist (no zero padding)."""
    x = np.asarray(x, float)
    n = max(int(n), 1)
    s = n // 2
    c = np.concatenate([[0.0], np.cumsum(x)])
    p = np.arange(len(x))
    lo, hi = np.clip(p - s, 0, len(x)), np.clip(p - s + n, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _acf_peak(seg, fs, lo_s=0.06, hi_s=0.5):
    seg = seg - seg.mean()
    if seg.std() == 0 or len(seg) < int(hi_s * fs) + 2:
        return 0.0
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    return float((ac / ac[0])[int(lo_s * fs):int(hi_s * fs)].max())


def validity_mask(x, fs, P=None):
    """Returns valid (bool per sample), reason (str per sample), low_amp (bool),
    segments [(start, stop)], noise_ptp (sensor-noise peak-to-peak estimate),
    blocks (DataFrame, one row per 1 s block: range, rel = range / median range, acf, decision)."""
    P = P or VALIDITY_PARAMS
    n = len(x)
    reason = np.full(n, "", dtype=object)
    reason[~np.isfinite(x)] = "missing"
    xf = np.where(np.isfinite(x), x, np.nan)
    w, h = max(int(P["block_s"] * fs), 2), max(int(P["block_hop_s"] * fs), 1)
    starts = np.arange(0, max(n - w, 0) + 1, h)
    r = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        b = xf[s:s + w]
        b = b[np.isfinite(b)]
        if len(b) > w // 2:
            r[k] = np.subtract(*np.percentile(b, [95, 5]))
    med = np.nanmedian(r)
    low_amp = np.zeros(n, bool)
    flat_blocks, blocks = [], []
    for k, s in enumerate(starts):
        if not np.isfinite(r[k]):
            continue
        rel, sl, acf, dec = r[k] / med, slice(s, s + w), np.nan, "ok"
        if rel > P["artifact_rel_range"]:
            pad = int(P["artifact_pad_s"] * fs)
            a0, a1 = max(0, s - pad), min(n, s + w + pad)
            reason[a0:a1] = np.where(reason[a0:a1] == "", "artifact", reason[a0:a1])
            dec = "artifact"
        elif rel < P["flat_rel_range"]:
            b = xf[sl][np.isfinite(xf[sl])]
            acf = _acf_peak(b, fs)
            if acf < P["flat_acf_max"]:
                reason[sl] = np.where(reason[sl] == "", "flat_or_dropout", reason[sl])
                flat_blocks.append(sl)
                dec = "flat_or_dropout"
            else:
                low_amp[sl] = True
                dec = "low_amp (periodic)"
        elif rel < P["low_amp_rel_range"]:
            low_amp[sl] = True
            dec = "low_amp"
        blocks.append(dict(start_s=s / fs, range=float(r[k]), median_range=float(med), rel=float(rel),
                           acf=acf, decision=dec))
    # noise floor: 250 ms pieces inside dropouts if any, else quietest 1% (capped at 1% of median range)
    q = max(int(0.25 * fs), 2)
    pieces = [np.ptp(xf[sl][i:i + q]) for sl in flat_blocks for i in range(0, w - q + 1, q)
              if np.all(np.isfinite(xf[sl][i:i + q]))]
    if pieces:
        noise_ptp = float(np.median(pieces))
    else:
        m = n // q
        pp = np.ptp(xf[:m * q].reshape(m, q), axis=1)
        noise_ptp = float(min(np.nanpercentile(pp, 1), 0.01 * med))
    segments = []
    for a, b in _runs(reason == ""):
        if b - a >= P["min_segment_s"] * fs:
            segments.append((int(a), int(b)))
        else:
            reason[a:b] = "short_segment"
    valid = reason == ""
    return valid, reason, low_amp & valid, segments, noise_ptp, pd.DataFrame(blocks)
'''))

cells.append(md("""## Cell B: peak selection

Scores every local maximum and keeps the ones that pass the prominence, width and spacing rules. Rejected candidates are kept with a reason.
"""))

cells.append(code('''# ============================================================================
# CELL B: peak selection (which local maxima count as breaths)
#   select_peaks() scores every local maximum of a signal and keeps those that pass the
#   prominence / width / spacing rules. Rejected candidates are returned with a reason.
# ============================================================================
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PEAK_PARAMS = dict(
    prominence_frac=0.20,      # required prominence = this x local robust range ...
    noise_mult=2.0,            # ... but at least this x noise floor
    local_range_win_s=1.0,     # window for the local range (s)
    local_range_pcts=(10, 90), # local range = p90 - p10
    prominence_wlen_s=1.0,     # prominence bases searched within +/- wlen/2 (None = whole segment)
    min_peak_width_s=0.015,    # half-prominence width; narrower = spike
    min_peak_distance_s=0.06,  # minimum spacing; the taller peak wins
)


def local_range(y, fs, win_s=1.0, pcts=(10, 90)):
    """Per-sample robust range (p_hi - p_lo) in a centred window, computed every 0.1 s."""
    n, w, hop = len(y), max(int(win_s * fs), 3), max(int(0.1 * fs), 1)
    centers = np.arange(0, n, hop)
    vals = [np.subtract(*np.percentile(y[max(0, c - w // 2):min(n, c + w // 2 + 1)], [pcts[1], pcts[0]]))
            for c in centers]
    return np.interp(np.arange(n), centers, vals)


def select_peaks(y, fs, noise_ptp, scale_signal=None, P=None):
    """y: baseline-corrected signal of one valid segment. scale_signal: where the local
    range is measured (default y). Returns (peak indices, candidate rows, local range).
    Candidate rows = every local maximum with prominence >= half the noise floor, each with
    its numbers and reason: 'counted', 'low_prominence', 'spike_width' or 'too_close'."""
    P = P or PEAK_PARAMS
    lr = local_range(y if scale_signal is None else scale_signal, fs, P["local_range_win_s"], P["local_range_pcts"])
    wlen = None if P["prominence_wlen_s"] is None else max(int(P["prominence_wlen_s"] * fs), 3)
    floor = P["noise_mult"] * noise_ptp
    cand, pr = find_peaks(y, prominence=0, wlen=wlen, width=0, rel_height=0.5)
    need = np.maximum(P["prominence_frac"] * lr[cand], floor)
    prom, width = pr["prominences"], pr["widths"] / fs
    ok_prom, ok_width = prom >= need, width >= P["min_peak_width_s"]
    surv = cand[ok_prom & ok_width]
    if len(surv):
        only = np.full(len(y), -np.inf)
        only[surv] = y[surv]
        sel, _ = find_peaks(only, distance=max(int(round(P["min_peak_distance_s"] * fs)), 1))
        sel = np.intersect1d(sel, surv)
    else:
        sel = surv
    keep = set(sel.tolist())
    rows = []
    for c, p, wd, nd, op, ow in zip(cand, prom, width, need, ok_prom, ok_width):
        counted = bool(op and ow and c in keep)
        if p < 0.5 * floor and not counted:
            continue
        why = "counted" if counted else ("low_prominence" if not op else ("spike_width" if not ow else "too_close"))
        rows.append(dict(idx=int(c), prominence=float(p), required=float(nd),
                         req_from_range=float(P["prominence_frac"] * lr[c]), req_from_noise=float(floor),
                         local_range=float(lr[c]), width_s=float(wd), reason=why))
    return np.asarray(sel, int), rows, lr
'''))

cells.append(md("""## Cell C: the detector

Runs on each valid segment: smoothing, a provisional pass to find cycles, a local baseline, the final pass, then a check that every pair of peaks has a real trough between them.
"""))

cells.append(code('''# ============================================================================
# CELL C: the detector (baseline, two passes, cycles)
#   detect_breaths() is what Sections 1-3 run on every recording. It uses cell A for
#   validity and cell B for peak selection.
# ============================================================================
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DETECTOR_PARAMS = dict(
    smooth_ms=10.0,           # running-mean smoothing (odd window, so no time shift)
    scale_highpass_hz=1.0,    # high-pass used ONLY for pass-A cycles + threshold scale (None = off)
    polarity=1,               # +1: inhalation is positive
)


def _highpass(seg, fs, hz):
    if hz is None:
        return seg
    b, a = butter(2, hz / (fs / 2.0), btype="high")
    return filtfilt(b, a, seg) if len(seg) > 27 else seg - seg.mean()


def _troughs_between(y, pk):
    return np.array([a + int(np.argmin(y[a:b])) for a, b in zip(pk[:-1], pk[1:])], int)


def _cycle_mean_baseline(x, pk):
    """Mean of x over each peak-to-peak cycle, at the cycle centre, interpolated
    (zero net airflow per cycle -> local zero-flow level)."""
    if len(pk) < 2:
        return np.full(len(x), np.nanmean(x))
    c = np.concatenate([[0.0], np.cumsum(x)])
    a, b = pk[:-1], pk[1:]
    return np.interp(np.arange(len(x)), (a + b) / 2.0, (c[b] - c[a]) / (b - a))


def _validate_cycles(y, pk, need_at, rows):
    """The trough between consecutive peaks must be at least the required prominence below
    the LOWER peak; otherwise that peak is dropped (no cycle without evidence)."""
    pk = list(pk)
    changed = True
    while changed and len(pk) > 1:
        changed = False
        for i, t in enumerate(_troughs_between(y, np.asarray(pk))):
            low = pk[i] if y[pk[i]] <= y[pk[i + 1]] else pk[i + 1]
            if y[low] - y[t] < need_at(low) - 1e-9:
                rows[:] = [r for r in rows if r["idx"] != low]          # replace its 'counted' row
                rows.append(dict(idx=int(low), prominence=float(y[low] - y[t]), required=float(need_at(low)),
                                 width_s=np.nan, reason="shallow_trough"))
                pk.remove(low)
                changed = True
                break
    return np.asarray(pk, int)


def detect_breaths(signal, fs, P=None):
    """Returns dict: peaks, troughs (sample indices), smoothed, baseline, valid, reason,
    low_amp, segments, noise_ptp, candidates (every candidate with its numbers and verdict),
    rejected (the non-counted candidates), blocks (Component A's per-block decisions)."""
    P = P or DETECTOR_PARAMS
    x = P["polarity"] * np.asarray(signal, float)
    n = len(x)
    valid, reason, low_amp, segments, noise_ptp, blocks = validity_mask(x, fs)
    sn = max(int(np.floor(fs / 1000.0 * P["smooth_ms"])), 1)
    sn += 1 - sn % 2
    xs, base = np.full(n, np.nan), np.full(n, np.nan)
    peaks, troughs, candidates = [], [], []
    for s0, s1 in segments:
        seg = running_mean(x[s0:s1], sn)
        xs[s0:s1] = seg
        hp = _highpass(seg, fs, P["scale_highpass_hz"])
        pkA, _, _ = select_peaks(hp, fs, noise_ptp)                      # pass A: provisional cycles
        bl = _cycle_mean_baseline(seg, pkA)
        y = seg - bl
        pk, rej, lr = select_peaks(y, fs, noise_ptp, scale_signal=hp)     # pass B: final peaks
        need_at = lambda i: max(PEAK_PARAMS["prominence_frac"] * lr[i], PEAK_PARAMS["noise_mult"] * noise_ptp)  # noqa: E731
        pk = _validate_cycles(y, pk, need_at, rej)
        if len(pk) >= 2:
            bl = _cycle_mean_baseline(seg, pk)                            # refine baseline, same peaks
            y = seg - bl
        base[s0:s1] = bl
        peaks.append(pk + s0)
        troughs.append(_troughs_between(y, pk) + s0 if len(pk) > 1 else np.array([], int))
        candidates += [dict(r, idx=r["idx"] + s0) for r in rej]
    cat = lambda v: np.concatenate(v).astype(int) if v else np.array([], int)  # noqa: E731
    cols = ["idx", "prominence", "required", "req_from_range", "req_from_noise", "local_range", "width_s", "reason"]
    cand = pd.DataFrame(candidates, columns=cols).sort_values("idx").reset_index(drop=True)
    return dict(peaks=cat(peaks), troughs=cat(troughs), smoothed=xs, baseline=base, valid=valid,
                reason=reason, low_amp=low_amp, segments=segments, noise_ptp=noise_ptp, blocks=blocks,
                candidates=cand, rejected=cand[cand.reason != "counted"].reset_index(drop=True),
                params=dict(PEAK_PARAMS))
'''))

cells.append(md("""## Section 1: Old vs new detector (Cohort 3 cagemate)

Windows from Sequoia's QC notebook where the old detector missed breaths.

- **Top, OLD:** legacy BreathMetrics. Counts a peak if `height > mean + std/2` of the **whole session**.
- **Bottom, NEW:** cells A-C. Counts a peak if `prominence >= max(0.20 x local 1 s range, 2 x noise floor)`, its half-width is at least 15 ms, and it is at least 60 ms from a taller peak.
- The table under each pair gives these numbers for every candidate. `[range]` or `[noise]` says which term set the requirement.
"""))

cells.append(code('''%matplotlib inline
import workbench_support as wb          # loading / plotting / tables only, no detection
import pandas as pd
pd.set_option("display.max_columns", 20)
'''))

PAIRS = [
    ("""### 1.1 Session 11_3, 300-301.5 s: breaths too small for the old cutoff
- **OLD: 0 peaks** (Sequoia's run: 4). Cutoff = `0.1 + 1428/2 = 714`. Transients up to ±35,000 inflate the std. The tallest breath here is 101.
- **NEW: 14 of 15 candidates count.** Each needs about 18 (20% of the local range). The rejected one (301.17 s, prominence 11) is a shoulder, not a cycle.""", 0),
    ("""### 1.2 Session 11_3, 550-551.5 s: same failure, bigger breaths
- **OLD: 0 peaks** (Sequoia's run: 15). Breaths are 45-261 high and the cutoff is still 714.
- **NEW: all 17 count.** Each clears its requirement (39-49) by 2.5-6x, so the count doesn't depend on the exact 20% setting.""", 1),
    ("""### 1.3 Session 11_3, 442-443.5 s: small breaths on a drift
- **OLD: 0 peaks.** The breaths ride a downward drift (heights -115 to 135).
- **NEW: 13 count.** Two close calls:
  - the bump at 442.19 s misses by 0.8 (22.6 vs 23.4);
  - at 443.3-443.5 s the local range drops to about 30, so the noise floor (6.9) sets the requirement.""", 2),
    ("""### 1.4 Session 10_3, 50-51.5 s: control, both methods agree
- **Both count 13.**
- No large transients, so the OLD cutoff is `0.1 + 422/2 = 211` and every breath (606-975) clears it.
- The NEW requirement here is 216-311, about 12x higher than in 11_3, because it scales with local breath size.""", 3),
    ("""### 1.5 Session 11_2, 490-491.5 s: signal loss
- **Both count 0.**
- OLD: nothing reaches 206.
- NEW: cell A marks the stretch invalid. The 1 s ranges are 5-9 against a session median of 1,116 (`rel ≈ 0.006 < 0.03`, ACF < 0.5).
- **Possible flaw:** two blocks just before this window (ACF 0.55-0.60) pass as "low_amp (periodic)" at 0.7% of normal amplitude. Over Sequoia's masked 482-516 s, this lets NEW count **65 peaks** (see 1.7).""", 4),
    ("""### 1.6 Session 11_2, 521-522.5 s: control after the dropout
- **Both count 15.**
- Breaths are 588-951 high (above 206). Prominences are 1,058-1,541, far above the requirement (228-264).""", 5),
    ("""### 1.7 Session 11_2, 499.8-501.3 s: where NEW can go wrong
Extra window, not in Sequoia's notebook.
- Blocks here are 2.6-4.8% of normal amplitude, but their ACF (0.59-0.79) passes cell A as periodic low-amplitude breathing.
- `0.20 x range` (5-8) is below the noise floor, so every requirement is the noise floor (11.4). **15 ripples count.**
- They may be noise, or real breathing through a slipping sensor.
- **To test a fix in cell A:** mark blocks below `flat_rel_range` as dead regardless of ACF, or raise `flat_rel_range` or `flat_acf_max`.""", 6),
]
for text, i in PAIRS:
    cells.append(md(text))
    cells.append(code(f"wb.plot_old_vs_new(detect_breaths, {i})"))

cells.append(md("""## Section 2: 40 review windows (Cohort 1)

4 s windows spread across the 30 Cohort 1 recordings. Bad signal is left out. Each panel title starts with its cohort.

| marker | meaning |
|---|---|
| blue ▲ | peak (counted) |
| purple ▼ | trough between two peaks |
| orange ● | near-miss: rejected, but reached 50% of the requirement |
| green ○ | peak your edit **added** vs the snapshot |
| red ✕ | peak your edit **removed** vs the snapshot |
| grey dashed line | local baseline |
| grey shading | cell A marked it invalid |

Look for clear cycles with no ▲ (missed breaths) and ▲ on something that isn't a full rise and fall (extra breaths).
"""))

cells.append(code('''%matplotlib inline
import workbench_support as wb          # loading / window choice / plotting only

COMPARE_TO = "v1"        # snapshot to compare against (see Section 4); None = no comparison
SHOW_LEGACY = False      # True: also draw the legacy BreathMetrics peaks (grey circles)

recs = wb.recordings()                   # cached 400 Hz signals, same loaders as the pipelines
bad = wb.bad_intervals(recs)            # frozen bad-signal screen -> workbench_bad_intervals.csv
windows = wb.select_windows(recs)        # frozen, evenly spread, good-signal windows
results = wb.run_all(detect_breaths, recs)
if COMPARE_TO and wb.load_snapshot(COMPARE_TO) is None:
    wb.save_snapshot(results, COMPARE_TO, params=dict(VALIDITY_PARAMS=VALIDITY_PARAMS, PEAK_PARAMS=PEAK_PARAMS,
                                                      DETECTOR_PARAMS=DETECTOR_PARAMS))
panel_table = wb.plot_windows(results, windows, compare_to=COMPARE_TO, show_legacy=SHOW_LEGACY)
'''))

cells.append(md("""## Section 3: Whole-recording peak counts vs the snapshot (Cohort 1)

All good signal, not just the 40 windows.
"""))

cells.append(code('''import pandas as pd
pd.set_option("display.max_rows", 60)
print("Panels with changes:", panel_table.query("added > 0 or removed > 0").panel.tolist())
print(f"Excluded as bad signal: {bad.dur_s.sum() / 60:.1f} min in {bad.key.nunique()} recordings "
      f"(details: workbench_bad_intervals.csv)")
wb.summary(results, compare_to=COMPARE_TO)
'''))

cells.append(md("""## Section 4: Save a snapshot

`v1` is the validated detector (2026-10-09). Don't overwrite it. To compare against a new snapshot, set `COMPARE_TO` in Section 2.
"""))

cells.append(code('''print("existing snapshots:", wb.list_snapshots())
# wb.save_snapshot(results, "v2", params=dict(VALIDITY_PARAMS=VALIDITY_PARAMS, PEAK_PARAMS=PEAK_PARAMS,
#                                             DETECTOR_PARAMS=DETECTOR_PARAMS))
'''))

cells.append(md("""## Section 5: Does the notebook still match the pipeline module?

`False` is expected after an edit. Copy the change into `../adaptive_breath_detection.py` before using it elsewhere.
"""))

cells.append(code('''from adaptive_breath_detection import detect_breaths as pipeline_detect
import numpy as np
key = sorted(recs)[0]
same = np.array_equal(pipeline_detect(recs[key][0], 400).breaths.peak.to_numpy().astype(int), results[key]["peaks"])
print(f"{key}: notebook peaks identical to pipeline module: {same}")
'''))

cells.append(md("""## Section 6: Bad signal per recording

Each recording is plotted whole. The shaded stretches are what Sections 2 and 3 leave out.

- **dead** (red): 1 s range below 10% of the recording median
- **artifact** (purple): range above 8x, or a spike above 5x
- **degraded** (orange): 5 s running median below 50%
- **buffer** (tan): the 10 s around dead or artifact stretches

Flags in the titles: **CONSIDER DROPPING** at 25% or more excluded, **review** at 10% or more, **WHOLE-SESSION LOW** when a recording's typical amplitude is below 25% of its group's.
"""))

cells.append(md("""### 6.1 Cohort 1 (Jun 2025): commented out

Uncomment a line to plot that group.
"""))

cells.append(code('''# wb.bad_signal_report(recs, group="aim1_cagemate", row_height=2.8, plots_only=True)
# wb.bad_signal_report(recs, group="aim1_blri", row_height=2.8, plots_only=True)
# wb.bad_signal_report(recs, group="aim1_ri1", row_height=2.8, plots_only=True)
# wb.bad_signal_report(recs, group="aim1_ri2", row_height=2.8, plots_only=True)
'''))

cells.append(md("""### 6.2 Cohort 2 (Jan 2026 BLA pilot): commented out

Not used by any analysis. The first run of a line loads that group's raw H5s.
"""))

cells.append(code('''# wb.bad_signal_report(wb.group_recordings("c2_cm"), group="c2_cm", row_height=2.0, plots_only=True)
# wb.bad_signal_report(wb.group_recordings("c2_ri1_d1"), group="c2_ri1_d1", row_height=2.0, plots_only=True)
# wb.bad_signal_report(wb.group_recordings("c2_ri1_d2"), group="c2_ri1_d2", row_height=2.0, plots_only=True)
# wb.bad_signal_report(wb.group_recordings("c2_ri2_d1"), group="c2_ri2_d1", row_height=2.0, plots_only=True)
# wb.bad_signal_report(wb.group_recordings("c2_ri2_d2"), group="c2_ri2_d2", row_height=2.0, plots_only=True)
'''))

cells.append(md("""### 6.3 Cohort 3 (Jul 2026): cagemate

11 recordings, including the 3 sessions in Section 1.
"""))

cells.append(code('''wb.bad_signal_report(wb.group_recordings("c3_cm"), group="c3_cm", row_height=2.2, plots_only=True)
'''))

cells.append(md("""### 6.4 Cohort 3 (Jul 2026): RI1 days 1-2

44 recordings, drawn as 4 figures.

- **Whole-session low:** 16 recordings (mice 9_1, 9_2, 9_4, 10_1, 10_4, 11_2, 13_4, 14_3, 14_4).
- `11_3_RI1_a_on_60711_094020` is missing its full date (probably 20260711).
"""))

cells.append(code('''wb.bad_signal_report(wb.group_recordings("c3_ri1"), group="c3_ri1", row_height=2.0, plots_only=True)
'''))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "biopipeline-env",
                                                                       "language": "python"}})
nbf.write(nb, "peak_detection_workbench.ipynb")
print("wrote peak_detection_workbench.ipynb")
