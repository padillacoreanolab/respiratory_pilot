"""
Visual check of behavior/respiration alignment: a short real-time clip with the
session video on top, a BORIS ethogram below it and the respiration trace at
the bottom, all on the respiration clock (bouts mapped through the video frame
index by cagemate_export.py). Dashed boxes show where the old method (BORIS
seconds used directly as respiration seconds) would have put each bout.

Run (biopipeline-env):  python alignment_clips.py <export_dir>
Clips go to <export_dir>/alignment_clips/ (not tracked by git).
"""
import os
import sys

import cv2
import h5py
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ekg_resp_to_h5 as E
from cagemate_export import LAB

# Two per cohort: one normal session, one whose mp4 fps differs from the capture.
CLIPS = [
    ("cohort1_aim1_CM", "baseline", "CM_s2_3_d2_4_20250623_151153_merged"),
    ("cohort1_aim1_CM", "baseline", "CM_s4_7_d4_8_20250623_193718_merged"),     # 23 fps mp4
    ("cohort2_bla_pilot_CM", "", "1_1_i_cm_1_3_s_20260107_163611"),
    ("cohort2_bla_pilot_CM", "", "3_1_i_cm_3_2_d_20260106_115157"),            # 24 fps mp4
    ("cohort3_bla_cohort2_CM", "", "11_4_d_CM_11_3_s_20260729_125055"),
    ("cohort3_bla_cohort2_CM", "", "13_2_d_CM_13_1_s_20260729_111038"),        # 16 fps mp4
]
CLIP_SEC = 20.0
COLORS = {"facial sniffing": "#d62728", "anogenital sniffing": "#9467bd", "body sniffing": "#ff7f0e",
          "digging": "#8c564b", "self-grooming": "#2ca02c", "allogrooming": "#17becf",
          "chasing": "#e377c2", "fighting": "#000000", "rearing": "#bcbd22", "Posturing": "#7f7f7f"}


def pick_window(beh, total_sec):
    """Densest CLIP_SEC of sniffing in the second half (where timing errors are largest)."""
    sniff = beh[beh["Behavior"].str.contains("sniff") & beh["start_resp_sec"].notna()]
    best, best_t = -1, total_sec / 2
    for t in np.arange(total_sec / 2, total_sec - CLIP_SEC, 1.0):
        cover = (sniff["stop_resp_sec"].clip(t, t + CLIP_SEC) - sniff["start_resp_sec"].clip(t, t + CLIP_SEC)).sum()
        if cover > best:
            best, best_t = cover, t
    return best_t


def make_clip(h5_path, mp4_path, beh_csv, out_mp4):
    beh = pd.read_csv(beh_csv)
    with h5py.File(h5_path, "r") as f:
        frame_time = f["video/frame_time_sec"][:]
        va = dict(f["video"].attrs)
        sig, sig_t = f["resp_clean/signal"][:], f["resp_clean/time"][:]
    mp4_fps, cap_fps = float(va["mp4_fps"]), float(va["fit_fps"])
    t0 = pick_window(beh, sig_t[-1])
    t1 = t0 + CLIP_SEC
    frames = np.flatnonzero((frame_time >= t0) & (frame_time <= t1))

    win = beh[(beh["stop_resp_sec"] > t0) & (beh["start_resp_sec"] < t1)]
    naive = beh[(beh["Stop (s)"] > t0) & (beh["Start (s)"] < t1)]
    rows = sorted(set(zip(win["Subject"], win["Behavior"])) | set(zip(naive["Subject"], naive["Behavior"])))
    row_of = {r: i for i, r in enumerate(rows)}

    fig = plt.figure(figsize=(8, 9.6), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, max(1.0, 0.28 * len(rows)), 1.3], hspace=0.3,
                          left=0.25, right=0.97, top=0.93, bottom=0.06)
    ax_v, ax_e, ax_r = (fig.add_subplot(gs[i]) for i in range(3))
    name = os.path.basename(h5_path)[:-3]
    fig.suptitle(f"{name}\ncapture {cap_fps:.2f} fps, mp4 encoded {mp4_fps:g} fps  |  clip plays in real time",
                 fontsize=9)

    cap = cv2.VideoCapture(mp4_path)
    for _ in range(frames[0]):           # sequential grab: frame-exact, unlike seeking
        cap.grab()
    ok, img = cap.read()
    im = ax_v.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_v.set_axis_off()
    label = ax_v.text(0.01, 0.98, "", transform=ax_v.transAxes, va="top", fontsize=8, color="w",
                      bbox=dict(facecolor="k", alpha=0.6, lw=0))

    for _, b in win.iterrows():
        y = row_of[(b["Subject"], b["Behavior"])]
        ax_e.broken_barh([(b["start_resp_sec"], b["duration_resp_sec"])], (y - 0.35, 0.7),
                         color=COLORS.get(b["Behavior"], "#444"))
    for _, b in naive.iterrows():
        y = row_of[(b["Subject"], b["Behavior"])]
        ax_e.broken_barh([(b["Start (s)"], b["Stop (s)"] - b["Start (s)"])], (y - 0.45, 0.9),
                         facecolor="none", edgecolor="0.3", ls="--", lw=0.8)
    ax_e.set_yticks(range(len(rows)))
    ax_e.set_yticklabels([f"{'subj' if s == 'subject' else 'agent'}: {b}" for s, b in rows], fontsize=7)
    ax_e.set_ylim(-0.6, len(rows) - 0.4)
    ax_e.set_title("BORIS: filled = aligned via frame index;  dashed = BORIS seconds used directly (old)",
                   fontsize=8)

    m = (sig_t >= t0) & (sig_t <= t1)
    ax_r.plot(sig_t[m], sig[m], lw=0.7, color="k")
    ax_r.set_ylabel("resp (a.u.)", fontsize=8)
    ax_r.set_xlabel("respiration time (s from start of recording)", fontsize=8)
    for ax in (ax_e, ax_r):
        ax.set_xlim(t0, t1)
        ax.tick_params(labelsize=7)
    cursors = [ax.axvline(t0, color="tab:blue", lw=1.2) for ax in (ax_e, ax_r)]

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    writer = imageio_ffmpeg.write_frames(out_mp4, (w, h), fps=round(cap_fps), codec="libx264",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=8)
    writer.send(None)
    for k in frames:
        if k != frames[0]:
            ok, img = cap.read()
            if not ok:
                break
            im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = frame_time[k]
        for c in cursors:
            c.set_xdata([t, t])
        label.set_text(f"frame {k}   BORIS/mp4 t = {k / mp4_fps:7.2f} s   resp t = {t:7.2f} s")
        fig.canvas.draw()
        writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
    writer.close()
    cap.release()
    plt.close(fig)
    return t0


def run(export_dir):
    out_dir = os.path.join(export_dir, "alignment_clips")
    os.makedirs(out_dir, exist_ok=True)
    for cohort, condition, name in CLIPS:
        man = pd.read_csv(os.path.join(export_dir, cohort, "manifest.csv")).set_index("session")
        rec = os.path.join(LAB, man.loc[name, "rec_file"])
        mp4 = E.find_videos(os.path.dirname(rec))[0]["mp4_path"]
        h5 = os.path.join(export_dir, cohort, condition, name + ".h5")
        beh = os.path.join(export_dir, cohort, "behavior_aligned", name + ".csv")
        out = os.path.join(out_dir, f"{cohort}__{name}.mp4")
        t0 = make_clip(h5, mp4, beh, out)
        print(f"{out}  (resp t {t0:.0f}-{t0 + CLIP_SEC:.0f} s)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "cagemate_exports")
