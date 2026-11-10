"""
Video timing, frame reading, alignment clips, and h264 -> mp4 conversion.

Frame timing comes from the H5 /video group written by
src/file_conversion/ekg_resp_to_h5.py (fitted Trodes frame times on the
respiration clock). See docs/DATA_README.md section 2.
"""
import os
import subprocess

import cv2
import h5py
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"facial sniffing": "#d62728", "anogenital sniffing": "#9467bd",
          "body sniffing": "#ff7f0e", "digging": "#8c564b", "self-grooming": "#2ca02c",
          "allogrooming": "#17becf", "chasing": "#e377c2", "fighting": "#000000",
          "rearing": "#bcbd22", "posturing": "#7f7f7f", "tails rattling": "#1f77b4"}


def load_video_timing(h5_path):
    """Per-frame respiration times + mp4 info from an H5's /video group."""
    with h5py.File(h5_path, "r") as f:
        if "video" not in f:
            raise ValueError(f"{h5_path} has no /video: re-convert with ekg_resp_to_h5.py")
        v = f["video"]
        frame_time = v["frame_time_sec"][:]
        resp_idx = v["frame_to_resp_index"][:]
        a = dict(v.attrs)
    return {
        "frame_time_sec": frame_time,            # s from /resp[0]
        "frame_in_resp": resp_idx >= 0,          # False = before/after resp, or corrupt
        "n_frames": int(a.get("mp4_n_frames", frame_time.size)),
        "mp4_fps": float(a["mp4_fps"]),          # encoded rate (what BORIS sees)
        "capture_fps": float(a["fit_fps"]),      # real camera rate
        "n_camera_gaps": int(a.get("n_camera_gaps", 0)),
    }


def read_frame(mp4_path, frame):
    """One RGB frame. Reads sequentially IOT get the exact frame (seeking can land off by a few)."""
    cap = cv2.VideoCapture(mp4_path)
    for _ in range(int(frame)):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


def plot_ethogram(beh, t0, t1, ax=None):
    """Bouts (resp-clock seconds) as colored bars, one row per subject + behavior."""
    ax = ax or plt.gca()
    win = beh[(beh["stop_resp_sec"] > t0) & (beh["start_resp_sec"] < t1)]
    rows = sorted(set(zip(win["Subject"], win["Behavior"])))
    for y, (s, b) in enumerate(rows):
        g = win[(win["Subject"] == s) & (win["Behavior"] == b)]
        ax.broken_barh(list(zip(g["start_resp_sec"], g["duration_resp_sec"])), (y - 0.35, 0.7),
                       color=COLORS.get(b, "#444"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{'subj' if s == 'subject' else 'agent'}: {b}" for s, b in rows], fontsize=7)
    ax.set_ylim(-0.6, max(len(rows), 1) - 0.4)
    ax.set_xlim(t0, t1)
    return ax


def make_alignment_clip(mp4_path, timing, beh, resp, resp_t, t0, dur, out_mp4):
    """Real-time mp4: video on top, ethogram, then respiration, with a moving cursor."""
    t1 = t0 + dur
    frames = np.flatnonzero((timing["frame_time_sec"] >= t0) & (timing["frame_time_sec"] <= t1))

    # Figure layout IOT show video, behavior and breathing on one time axis
    fig = plt.figure(figsize=(7.2, 8), dpi=100)  # 720x800 px: divisible by 8 for the encoder
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.3,
                          left=0.27, right=0.97, top=0.95, bottom=0.07)
    ax_v, ax_e, ax_r = (fig.add_subplot(gs[i]) for i in range(3))
    im = ax_v.imshow(read_frame(mp4_path, frames[0]))
    ax_v.set_axis_off()
    plot_ethogram(beh, t0, t1, ax_e)
    m = (resp_t >= t0) & (resp_t <= t1)
    ax_r.plot(resp_t[m], resp[m], lw=0.7, color="k")
    ax_r.set_xlim(t0, t1)
    ax_r.set_xlabel("respiration time (s)")
    cursors = [ax.axvline(t0, color="tab:blue") for ax in (ax_e, ax_r)]

    # Write at the capture rate IOT play the clip in real time
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    writer = imageio_ffmpeg.write_frames(out_mp4, (w, h), fps=round(timing["capture_fps"]),
                                         codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=8)
    writer.send(None)
    cap = cv2.VideoCapture(mp4_path)
    for _ in range(frames[0]):
        cap.grab()
    for k in frames:
        ok, img = cap.read()
        if not ok:
            break
        im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for c in cursors:
            c.set_xdata([timing["frame_time_sec"][k]] * 2)
        fig.canvas.draw()
        writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
    writer.close()
    cap.release()
    plt.close(fig)
    return out_mp4


def convert_h264_to_mp4(h264_path, fps, out_path=None, duration_s=None):
    """
    Raw Trodes .h264 -> SLEAP/BORIS-friendly mp4 (from src/vid_conversion.ipynb).
    Pass the capture fps IOT avoid ffmpeg's 25 fps guess for raw h264, which
    makes mp4 seconds differ from real seconds. duration_s converts only the
    first N seconds (for quick tests).
    """
    out_path = out_path or h264_path.replace(".h264", "_converted.mp4")
    if os.path.exists(out_path):
        return out_path
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", str(fps), "-i", h264_path]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "superfast", "-crf", "23", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
