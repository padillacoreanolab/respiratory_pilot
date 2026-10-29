"""Check camera vs. Trodes sync: headstage accelerometer (Trodes clock) vs. SLEAP body speed (video clock).

Both signals measure how much the subject is moving. If video frame times are right, the two movement
envelopes line up at lag 0; a lag L > 0 means the video is late (frame times too early) by L seconds.
Accelerometer: |a - 0.5 Hz low-pass(a)| over X/Y/Z, 200 ms smoothed. Video: SLEAP subject centroid speed,
200 ms smoothed. Lag = peak of the normalized cross-correlation (+/-1 s, 10 ms steps), computed for the whole
session and for each half (a change between halves = clock drift).
Frame times: raw .videoTimeStamps (as in the shared H5) and a straight-line fit per unbroken run (as in
src/file_conversion/ekg_resp_to_h5.py).

Input: outputs/r34_shared/manifest.json, D:/R34_shared_h5 (SLEAP + frame times), accelerometer .dat exports.
Output: outputs/r34_shared/video_sync_imu/ (sync_lags.csv, sync_summary.png).
Usage (biopipeline-env): python scripts/check_video_sync_imu.py
"""
from pathlib import Path
import sys, json
import numpy as np, h5py, pandas as pd
from scipy.signal import butter, sosfiltfilt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
from convert_shared_h5 import trodes, OUT as H5DIR

OUT = ROOT/'outputs/r34_shared/video_sync_imu'
FS = 100                      # analysis grid, Hz
MAXLAG = 100                  # samples = 1 s
SMOOTH = 20                   # samples = 200 ms


def smooth(x, n=SMOOTH):
    return np.convolve(x, np.ones(n)/n, mode='same')


def fit_frames(ft):
    """Straight-line fit of frame time vs frame index, separately for each unbroken run."""
    d = np.diff(ft); breaks = np.flatnonzero(d > 5*np.median(d)) + 1; out = ft.copy()
    for seg in np.split(np.arange(len(ft)), breaks):
        if len(seg) >= 2: out[seg] = np.polyval(np.polyfit(seg, ft[seg], 1), seg)
    return out


def xcorr(a, v, ok):
    """Normalized correlation of a with v shifted by each lag (v[i+lag] vs a[i]); NaN-safe via mask ok."""
    lags = np.arange(-MAXLAG, MAXLAG+1); r = np.full(len(lags), np.nan)
    for k, L in enumerate(lags):
        i = np.arange(max(0, -L), min(len(a), len(a)-L))
        m = ok[i] & ok[i+L]
        if m.sum() > 1000: r[k] = np.corrcoef(a[i][m], v[i+L][m])[0, 1]
    return lags/FS, r


def accel_envelope(rec_dir, origin, fs, grid):
    axes = []
    for ax in 'XYZ':
        f = sorted(Path(rec_dir).rglob(f'*analog_Headstage_Accel{ax}.dat'))
        if len(f) != 1: return None
        a, _ = trodes(f[0], '<i2'); tk, _ = trodes(sorted(f[0].parent.glob('*timestamps.dat'))[0], '<u4')
        t = (tk.astype('f8') - origin)/fs
        g = a[np.clip(np.searchsorted(t, grid), 0, len(a)-1)].astype(float)   # sample-and-hold onto the grid
        axes.append(g - sosfiltfilt(butter(2, .5, 'low', fs=FS, output='sos'), g))   # remove gravity / slow tilt
    return smooth(np.sqrt(np.sum(np.square(axes), axis=0)))


def video_envelope(h, ft, grid, quantity='accel'):
    """SLEAP subject centroid movement on the grid: 'accel' = |change in velocity| (same quantity the
    accelerometer measures, so no physical lead/lag), 'speed' = |velocity|."""
    sg = h['behavior/sleap']; names = [n.decode() if isinstance(n, bytes) else n for n in sg['track_names'][:]]
    if 'Subject' not in names: return None
    xy = sg['pose_xy'][:, names.index('Subject')]                # frames x nodes x 2
    with np.errstate(all='ignore'):
        c = np.nanmean(xy, axis=1)                               # centroid of found nodes
        dt = np.diff(ft); vel = np.diff(c, axis=0)/dt[:, None]   # velocity between frames i and i+1
        if quantity == 'speed':
            sp = np.r_[np.nan, np.hypot(*vel.T)]
        else:                                                    # centered: frame i uses velocities i-1 and i
            sp = np.r_[np.nan, np.hypot(*np.diff(vel, axis=0).T)/((dt[1:]+dt[:-1])/2), np.nan]
    ok = np.isfinite(sp); v = np.interp(grid, ft[ok], sp[ok])
    gap = np.interp(grid, ft, (~ok).astype(float)) > 0            # grid points next to missing tracking
    v[gap] = np.nan
    return smooth(np.nan_to_num(v)), ~gap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = [r for r in json.load(open(ROOT/'outputs/r34_shared/manifest.json')) if r['experiment'] == 'Valence' and r['sleap']]
    rows, curves = [], {}
    for r in man:
        p = H5DIR/'Valence_cohort_1'/f"{r['session_id']}.h5"
        if not p.exists(): print('skip (no shared H5):', r['session_id']); continue
        with h5py.File(p, 'r') as h:
            if 'behavior/sleap' not in h or 'frame_time_s' not in h['behavior/sleap']: print('skip (SLEAP frames do not match video):', r['session_id']); continue
            origin = int(h['respiration'].attrs['recording_tick_origin']); fs = float(h['respiration'].attrs['clock_rate_hz'])
            ft_raw = h['video/frame_time_raw_s' if 'frame_time_raw_s' in h['video'] else 'video/frame_time_s'][:]
            # only where both video and Trodes data exist (e.g. RI2 3.5: Trodes stopped at 3:33, video ran 10 min)
            grid = np.arange(max(ft_raw[0], 0)+2, min(ft_raw[-1], h['respiration/preprocessed/time_s'][-1])-2, 1/FS)
            acc = accel_envelope(r['rec_dir'], origin, fs, grid)
            if acc is None: print('skip (no accelerometer export):', r['session_id']); continue
            for kind, ft, q in [('raw', ft_raw, 'accel'), ('fitted', fit_frames(ft_raw), 'accel'), ('fitted_speed', fit_frames(ft_raw), 'speed')]:
                ve = video_envelope(h, ft, grid, q)
                if ve is None: print('skip (no Subject track):', r['session_id']); break
                v, ok = ve; ok &= np.isfinite(acc)
                lag, cc = xcorr(acc, v, ok); k = np.nanargmax(cc)
                half = len(grid)//2; hl = []
                for sl in (slice(0, half), slice(half, None)):
                    l2, c2 = xcorr(acc[sl], v[sl], ok[sl]); hl.append(l2[np.nanargmax(c2)])
                rows.append(dict(session=r['session_id'], phase=r['phase'], frames=kind, lag_s=lag[k], peak_r=cc[k],
                                 r_at_0=cc[MAXLAG], lag_first_half_s=hl[0], lag_second_half_s=hl[1],
                                 tracked_fraction=float(ok.mean())))
                curves[(r['session_id'], kind)] = (lag, cc)
                print(f"{r['session_id'][:24]:24s} {kind:6s} lag {lag[k]*1000:+5.0f} ms  r {cc[k]:.2f}  halves {hl[0]*1000:+.0f}/{hl[1]*1000:+.0f} ms", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT/'sync_lags.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for (sid, kind), (lag, cc) in curves.items():
        if kind == 'fitted': axs[0].plot(lag*1000, cc, lw=.8, alpha=.7)
    axs[0].axvline(0, color='k', lw=.6); axs[0].set_xlim(-500, 500)
    axs[0].set(xlabel='video shift (ms); + = video late', ylabel='correlation', title='Accelerometer vs SLEAP acceleration, each recording (fitted frame times)')
    f = df[df.frames == 'fitted']
    axs[1].scatter(f.lag_first_half_s*1000, np.arange(len(f)), label='first half', marker='<')
    axs[1].scatter(f.lag_second_half_s*1000, np.arange(len(f)), label='second half', marker='>')
    axs[1].scatter(f.lag_s*1000, np.arange(len(f)), color='k', label='whole session')
    axs[1].set_yticks(np.arange(len(f))); axs[1].set_yticklabels([s[:12] for s in f.session], fontsize=8)
    axs[1].axvline(0, color='k', lw=.6); axs[1].axvspan(-67/2, 67/2, color='0.9', zorder=0, label='± half a frame (15 fps)')
    axs[1].set(xlabel='best lag (ms); + = video late', title='Best lag per recording'); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'sync_summary.png', dpi=110)
    print('->', OUT)


if __name__ == '__main__':
    main()
