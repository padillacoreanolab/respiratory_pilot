"""Real-BORIS misalignment between camera/BORIS and respiration in Aim1 merged recordings.

For each BORIS event:
  old   : respiration sample = round(Start(s) * 20000) in the merged array (old pipeline)
  true  : Trodes tick of the BORIS frame (Image index start) from .videoTimeStamps
  error = (tick of old sample - tick of true frame) / 20000   [s]; positive = old reads respiration too late
"""
import os, sys, glob, re
import numpy as np, pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, r"C:\Users\thoma\Code\ResearchCode\respiratory_pilot\notebooks\thomas_notebooks\resp_classification")
import resp_helper_functions as rhf

HERE = os.path.dirname(os.path.abspath(__file__))
REC = r"C:\Users\thoma\UFL Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\Baseline cagemate interactions"
BOR = r"C:\Users\thoma\UFL Dropbox\Thomas Heeps\Padilla-Coreano Lab\2025\ECG_cohort1\Aim1\AIM1\Day1_new\CM interactions\cm_boris\baseline csv"
EXCLUDE = {"CM_s4_8_sub4_7_20250623.csv"}  # same exclusion as build_interaction_boris_catalog
SNIFF = {"facial sniffing", "body sniffing", "anogenital sniffing"}
FS, FS_DS = 20000, 400


def load(p, dt):
    raw = open(p, "rb").read(); i = raw.find(b"<End settings>") + len(b"<End settings>")
    while raw[i:i + 1] in (b"\n", b"\r"): i += 1
    return np.frombuffer(raw[i:], dt)


def breath_rate(sig, fs, grid_fs=20):
    """Instantaneous breathing rate (Hz) from inhalation peaks, on a regular grid."""
    s = sig - np.median(sig)
    mad = np.median(np.abs(s)) * 1.4826
    pk, _ = find_peaks(s, distance=int(fs / 20), prominence=0.5 * mad)
    t = pk[1:] / fs; r = fs / np.diff(pk)
    g = np.arange(0, len(sig) / fs, 1 / grid_fs)
    return g, np.interp(g, t, r, left=np.nan, right=np.nan)


# pick one BORIS file per session (non-VT preferred, exclusion respected)
files = {}
for p in sorted(glob.glob(os.path.join(BOR, "*.csv"))):
    b = os.path.basename(p)
    if b in EXCLUDE: continue
    key = re.match(r"(CM_.+?_\d{8}(?:_\d{6})?)", b).group(1)
    if key not in files or "_VT" in os.path.basename(files[key]):
        files[key] = p

err_rows, eta, eta_sess = [], {"old": [], "new": []}, {"old": [], "new": []}
W = 6.0  # s around onset
for key, bp in files.items():
    sess = glob.glob(os.path.join(REC, key + "*.rec"))
    sess = [s for s in sess if os.path.isdir(s)]
    if not sess: print("no rec for", key); continue
    S = sess[0]
    ts = load(glob.glob(os.path.join(S, "*merged.analog", "*timestamps.dat"))[0], np.uint32).astype(np.int64)
    resp = load(glob.glob(os.path.join(S, "*merged.analog", "*ECU_Ain1.dat"))[0], np.int16).astype(np.float64)
    vts = load(glob.glob(os.path.join(S, "*.videoTimeStamps"))[0], np.uint32).astype(np.int64)
    n = np.arange(vts.size); a, b0 = np.polyfit(n, vts, 1); vfit = a * n + b0
    keep = np.r_[True, np.diff(ts) != 0]
    ts_d, resp_d = ts[keep], resp[keep]

    df = pd.read_csv(bp)
    if df.empty:
        print(f"{key}: BORIS file is empty, skipped"); continue
    df = df[df["Behavior"].str.lower().str.strip().isin(SNIFF) & (df["Subject"].str.lower().str.strip() == "subject")]
    df = df.dropna(subset=["Start (s)", "Image index start"])
    fps_boris = float(pd.read_csv(bp)["FPS (frame/s)"].iloc[0])

    # per-event exact error
    for _, ev in df.iterrows():
        i_old = int(round(ev["Start (s)"] * FS))
        true_tick = vfit[min(int(ev["Image index start"]), vfit.size - 1)]
        old_tick = ts[i_old] if i_old < ts.size else np.nan
        err_rows.append(dict(session=key, boris_fps=fps_boris, t_video=ev["Start (s)"],
                             t_true=(true_tick - ts[0]) / FS, err=(old_tick - true_tick) / FS))

    # breathing rate under both alignments (user's preprocessing, then peak-based rate)
    sig_old, _ = rhf.preprocess_resp_to_target_rate(resp, FS, FS_DS)    # merged, copies kept (old)
    sig_new, _ = rhf.preprocess_resp_to_target_rate(resp_d, FS, FS_DS)  # copies removed (new)
    g_old, r_old = breath_rate(sig_old, FS_DS); g_new, r_new = breath_rate(sig_new, FS_DS)
    lag = np.arange(-W, W + 1e-9, 0.05)
    for _, ev in df.iterrows():
        t_old = ev["Start (s)"]                                                  # old: BORIS seconds on index axis
        t_new = (vfit[min(int(ev["Image index start"]), vfit.size - 1)] - ts_d[0]) / FS  # new: frame -> tick -> sample
        for lab, g, r, t0 in (("old", g_old, r_old, t_old), ("new", g_new, r_new, t_new)):
            if t0 - W < 0 or t0 + W > g[-1]: continue
            eta[lab].append(np.interp(t0 + lag, g, r)); eta_sess[lab].append(key)
    print(f"{key}: {len(df)} subject sniff onsets, BORIS fps {fps_boris}")

err = pd.DataFrame(err_rows); err.to_csv(os.path.join(HERE, "misalign_events.csv"), index=False)
np.savez(os.path.join(HERE, "misalign_eta.npz"), lag=np.arange(-W, W + 1e-9, 0.05),
         old=np.array(eta["old"]), new=np.array(eta["new"]),
         old_sess=np.array(eta_sess["old"]), new_sess=np.array(eta_sess["new"]))
print(err.groupby("session")["err"].describe()[["count", "min", "50%", "max"]].round(2))
