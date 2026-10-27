"""Read-only inventory of raw recordings, h5s and BORIS files for the three respiration cohorts."""
import os, re, glob, json
import pandas as pd

UF = r"C:\Users\thoma\UF Dropbox\Thomas Heeps\Padilla-Coreano Lab"
A1 = os.path.join(UF, r"2025\ECG_cohort1\Aim1\AIM1\Day1_new")
P1 = os.path.join(UF, r"2025\BLA_ChR_resp_pilot1 (SS)")
C2 = os.path.join(UF, r"2026\BLA_ChR_resp_cohort2")
REPO_H5 = r"C:\Users\thoma\Code\ResearchCode\respiratory_pilot\src\file_conversion\h5_outputs"

# (cohort, experiment, raw dir or None, [h5 dirs], [boris dirs], name filter regex for shared h5 dirs)
EXPS = [
    ("1 Aim1 (ECG_cohort1)", "Cagemate baseline", os.path.join(A1, r"CM interactions\Baseline cagemate interactions"),
     [os.path.join(A1, r"CM interactions\cm_h5\baseline_cagemate_interactions_h5_outputs")], [os.path.join(A1, r"CM interactions\cm_boris")], r"^CM_(?!.*iso)"),
    ("1 Aim1 (ECG_cohort1)", "Cagemate 1-day isolation", os.path.join(A1, r"CM interactions\1_day_isolation"),
     [os.path.join(REPO_H5, "cagemate_day_1_h5_outputs")], [os.path.join(A1, r"CM interactions\cm_boris")], r"1d_i"),
    ("1 Aim1 (ECG_cohort1)", "Cagemate 10-day isolation", os.path.join(A1, r"CM interactions\10_day_isolation"),
     [os.path.join(REPO_H5, "cagemate_day_10_h5_outputs")], [os.path.join(A1, r"CM interactions\cm_boris")], r"10d_iso"),
    ("1 Aim1 (ECG_cohort1)", "Valence BLRI (baseline)", None, [os.path.join(A1, "Resp_h5")], [os.path.join(A1, "boris_csv_RI1 and RI2")], r"^BLRI_"),
    ("1 Aim1 (ECG_cohort1)", "Valence RI1 (positive)", None, [os.path.join(A1, "Resp_h5")], [os.path.join(A1, "boris_csv_RI1 and RI2")], r"^RI1_"),
    ("1 Aim1 (ECG_cohort1)", "Valence RI2 (negative)", None, [os.path.join(A1, "Resp_h5")], [os.path.join(A1, "boris_csv_RI1 and RI2")], r"^RI2_"),
    ("2 BLA pilot 1 (Jan 2026)", "Cagemate home cage", os.path.join(P1, "BLA_resp_ChR_cm_hc"),
     [os.path.join(P1, r"BLA_resp_ChR_cm_hc\h5_outputs")], [os.path.join(P1, "BLA_resp_ChR_cm_boris")], None),
    ("2 BLA pilot 1 (Jan 2026)", "Baseline (bl_cc)", os.path.join(P1, r"BLA_ChR_resp_bl_cc\BLA_resp_ChR_bl_cc"), [], [], None),
    ("2 BLA pilot 1 (Jan 2026)", "RI1 day 1", os.path.join(P1, "BLA_ChR_resp_RI1_hc_day1"),
     [os.path.join(P1, r"BLA_ChR_resp_RI1_hc_day1\h5_outputs")], [os.path.join(P1, "RI1_day1_boris")], None),
    ("2 BLA pilot 1 (Jan 2026)", "RI1 day 2", os.path.join(P1, "BLA_ChR_resp_RI1_hc_day2"),
     [os.path.join(P1, r"BLA_ChR_resp_RI1_hc_day2\h5_outputs")], [os.path.join(P1, "RI1_day2_boris")], None),
    ("2 BLA pilot 1 (Jan 2026)", "RI2 post day 1", os.path.join(P1, "BLA_ChR_resp_RI2_hc_post_d1"),
     [os.path.join(P1, r"BLA_ChR_resp_RI2_hc_post_d1\h5_outputs")], [os.path.join(P1, "RI2_day1_boris_post")], None),
    ("2 BLA pilot 1 (Jan 2026)", "RI2 post day 2", os.path.join(P1, "BLA_ChR_resp_RI2_hc_post_d2"),
     [os.path.join(P1, r"BLA_ChR_resp_RI2_hc_post_d2\h5_outputs")], [os.path.join(P1, "RI2_day2_boris_post")], None),
    ("2 BLA pilot 1 (Jan 2026)", "OFT", os.path.join(P1, "BLA_ChR_resp_OFT"),
     [os.path.join(P1, r"BLA_ChR_resp_OFT\BLA_OFT_h5_outputs")], [], None),
    ("2 BLA pilot 1 (Jan 2026)", "wrong_pulse", os.path.join(P1, "wrong_pulse"), [], [], None),
    ("3 BLA cohort 2 folder (Jul 2026)", "Cagemate home cage", os.path.join(C2, "BLA_ChR_resp_CM_hc"),
     [os.path.join(C2, r"BLA_ChR_resp_CM_hc\h5_outputs")], [os.path.join(C2, r"BLA_ChR_resp_CM_hc\BORIS_csv")], None),
    ("3 BLA cohort 2 folder (Jul 2026)", "RI1 day 1+2", os.path.join(C2, "BLA_ChR_resp_RI1_hc_day1_2"),
     [os.path.join(C2, r"BLA_ChR_resp_RI1_hc_day1_2\h5_outputs")], [os.path.join(C2, r"BLA_ChR_resp_RI1_hc_day1_2\BORIS_csv")], None),
    ("3 BLA cohort 2 folder (Jul 2026)", "OFT", os.path.join(C2, "OFT"), [], [], None),
]

DT = re.compile(r"(\d{8}_\d{6})")


def core(name):
    """Recording key: everything up to the date_time stamp, normalized."""
    n = os.path.basename(name)
    for suf in (".analysis.h5", ".h5", ".rec.csv", ".csv", ".rec", ".slp"):
        if n.endswith(suf): n = n[: -len(suf)]
    n = n.replace("_merged", "")
    m = DT.search(n)
    return (n[: m.end()] if m else n).lower()


def loose(k):
    """Fallback key: drop date/time and scorer/version tags, then all punctuation."""
    k = k.lower()
    k = re.split(r"_\d{7,}|_vt|_boris|_rescore|_new|_heeps|_ss|recfinal|\.1", k)[0]
    return re.sub(r"[^a-z0-9]", "", k)


def stamp(k):
    m = DT.search(k); return m.group(1) if m else None


def raw_status(rec_dir):
    files = os.listdir(rec_dir)
    recs = [f for f in files if f.endswith(".rec") and os.path.isfile(os.path.join(rec_dir, f))]
    merged = any(f.endswith("merged.rec") for f in recs)
    an = [d for d in glob.glob(os.path.join(rec_dir, "*.analog")) if os.path.isdir(d)]
    ecu = bool(an) and any("ECU_Ain1" in f for d in an for f in os.listdir(d))
    ts = bool(an) and any(f.endswith("timestamps.dat") for d in an for f in os.listdir(d))
    vts = any(f.endswith(".videoTimeStamps") for f in files)
    mp4 = any(f.lower().endswith(".mp4") for f in files)
    parts = []
    if not recs: parts.append("no .rec")
    if not ecu: parts.append("no ECU_Ain1 export")
    elif not ts: parts.append("no timestamps.dat")
    if not vts: parts.append("no videoTimeStamps")
    if not mp4: parts.append("no mp4")
    return ("merged" if merged else "plain" if recs else "-"), ("complete" if not parts else "; ".join(parts))


def h5_format(path):
    import h5py
    try:
        with h5py.File(path, "r") as f:
            k = set(f.keys())
    except Exception as e:
        return f"unreadable ({type(e).__name__})"
    if "timestamps" in k and "video" in k: return "NEW v2 (timestamps + video)"
    if "resp_clean" in k: return "old: analog/dio/time + resp_clean"
    if "analog" in k: return "old: analog/dio/time (ECU only)"
    if "resp" in k: return "old: top-level resp (no timestamps)"
    return "old: NO resp (keys: " + ",".join(sorted(k)) + ")"


rows, fmt_cache = [], {}


def rid(k):
    """Recording identity: the date_time stamp when present (unique per session), else the name."""
    return stamp(k) or k


for cohort, exp, raw, h5dirs, bdirs, filt in EXPS:
    recs, h5s, names = {}, {}, {}
    if raw and os.path.isdir(raw):
        for d in sorted(glob.glob(os.path.join(raw, "*.rec"))):
            if os.path.isdir(d):
                recs[rid(core(d))] = d; names.setdefault(rid(core(d)), {})["raw"] = core(d)
    for hd in h5dirs:
        if not os.path.isdir(hd): continue
        for p in sorted(glob.glob(os.path.join(hd, "*.h5"))):
            if filt and not re.search(filt, os.path.basename(p)): continue
            i = rid(core(p))
            if i not in h5s:
                h5s[i] = p; names.setdefault(i, {})["h5"] = core(p)
    bor = {}
    for bd in bdirs:
        if not os.path.isdir(bd): continue
        for p in sorted(glob.glob(os.path.join(bd, "**", "*.csv"), recursive=True)):
            if filt and not re.search(filt, os.path.basename(p)): continue
            bor.setdefault(core(p), []).append(os.path.basename(p))
    used = set()
    for i in sorted(set(recs) | set(h5s)):
        nm = names[i]; disp = nm.get("raw") or nm.get("h5")
        rtype, rstat = raw_status(recs[i]) if i in recs else ("-", "raw NOT in Dropbox" if raw is None else "no raw folder")
        hp = h5s.get(i); hfmt = ""
        if hp:
            hd = os.path.dirname(hp)
            if hd not in fmt_cache: fmt_cache[hd] = h5_format(hp)
            hfmt = fmt_cache[hd]
        match = [kk for kk in bor if kk in nm.values() or (stamp(kk) and stamp(kk) == stamp(disp))]
        if not match:
            match = [kk for kk in bor if any(loose(kk) == loose(v) for v in nm.values())]
        used.update(match)
        b = [x for kk in match for x in bor[kk]]
        note = ""
        if "raw" in nm and "h5" in nm and nm["raw"] != nm["h5"]:
            note = f"NAME MISMATCH: raw '{nm['raw']}' vs h5 '{nm['h5']}'"
        rows.append(dict(cohort=cohort, experiment=exp, recording=disp, raw=rtype, raw_status=rstat,
                         h5="yes" if hp else "NO", h5_format=hfmt,
                         h5_folder=os.path.dirname(hp).replace(UF, "…").replace(REPO_H5, "repo h5_outputs") if hp else "",
                         boris="; ".join(b), note=note))
    unmatched = sorted(set(bor) - used)
    if unmatched:
        rows.append(dict(cohort=cohort, experiment=exp, recording="(BORIS files with no matching recording)", raw="", raw_status="",
                         h5="", h5_format="", h5_folder="", boris="; ".join(x for u in unmatched for x in bor[u]), note=""))

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.csv")
df.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
summ = df[df.recording.str.startswith("(") == False].groupby(["cohort", "experiment"], sort=False).agg(
    recordings=("recording", "size"), raw_complete=("raw_status", lambda s: (s == "complete").sum()),
    h5=("h5", lambda s: (s == "yes").sum()), boris=("boris", lambda s: (s != "").sum()))
print(summ.to_string())
print("\nh5 formats per folder:"); [print(" ", k.replace(UF, "…"), "->", v) for k, v in fmt_cache.items()]
print("\nraw problems:"); print(df[(df.raw_status != "complete") & ~df.raw_status.isin(["raw NOT in Dropbox", ""])][["experiment", "recording", "raw_status"]].to_string())
print("\nunmatched BORIS:"); print(df[df.recording.str.startswith("(")][["experiment", "boris"]].to_string())
