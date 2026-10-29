"""Build a self-contained HTML page for manual review of local-prominence peaks.

Nothing is masked: the whole recording is shown and searched for candidates, and the
reviewer marks unusable stretches by hand. The automatic spike mask and proposed dropouts
are passed only as optional hints. All candidates from src/rsp/local_peaks.py are exported
with their prominence and relative (12% local span) threshold, so changing the ADC floor in
the browser gives the same peaks as detect_local_peaks(minimum_prominence=floor).
Never modifies H5 files.

Two methods (--method):
  original100 (default): the lab's original 100 Hz processing and peak finder (src/rsp/original_method.py);
                         floor 0 reproduces it exactly. Page: manual_peak_review/review_pages_100Hz/<mouse>.html
  local400:              400 Hz local-prominence detector (src/rsp/local_peaks.py). Page: manual_peak_review/old_400Hz_pages/<mouse>.html

Usage (biopipeline-env):
  python scripts/build_peak_review_html.py --mouse 11_3
  python scripts/build_peak_review_html.py --mouse 11_3 --method local400 --prominence 25
"""
from pathlib import Path
import sys, json, base64, argparse
import numpy as np, h5py
from scipy.signal import find_peaks
from scipy.ndimage import percentile_filter, maximum_filter1d

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from src.rsp.local_peaks import detect_local_peaks
from src.rsp.signal_quality import propose_dropouts
from review_peaks_gui import load_unmasked, find_h5, OUT, FS
from src.rsp.original_method import original_signal, original_candidates, FS as OFS

RELATIVE, MIN_INTERVAL_S = .12, .04


def candidates(y, bad, fs):
    """detect_local_peaks with the ADC floor left open; same segmentation, spacing and edge trim."""
    bad = bad | ~np.isfinite(y)
    edges = np.diff(np.r_[False, ~bad, False].astype(int))
    rows = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b-a < fs: continue
        x = y[a:b]; size = int(fs) | 1
        span = percentile_filter(x, 90, size=size, mode='nearest')-percentile_filter(x, 10, size=size, mode='nearest')
        ix, props = find_peaks(x, prominence=0, distance=max(1, int(fs*MIN_INTERVAL_S)), wlen=size)
        keep = (ix >= int(.1*fs))&(ix < len(x)-int(.1*fs))
        rows += zip(ix[keep]+a, props['prominences'][keep], RELATIVE*span[ix[keep]])
    s, p, r = map(np.asarray, zip(*rows))
    return s.astype(int), p, r


def spike_flag(y, fs):
    """Hint only: >20 robust SD from the median, padded 3 s each side (same rule as the 400 Hz page)."""
    med = np.median(y); sc = 1.4826*np.median(np.abs(y-med))
    return maximum_filter1d((np.abs(y-med) > 20*sc).astype('u1'), size=6*int(fs)+1) > 0


def build_local400(mouse, floor):
    path, t, y, auto_spike = load_unmasked(mouse)
    s, prom, rel = candidates(y, np.zeros(len(y), bool), FS)
    for f in (floor, 15.):  # browser filtering must reproduce the Python detector
        ref = detect_local_peaks(y, FS, None, minimum_prominence=f)[0]
        assert np.array_equal(np.sort(ref), s[prom >= np.maximum(f, rel)]), f
    with h5py.File(path, 'r') as h: raw = h['breathmetrics/pressure_preprocessed'][:].astype(float)
    drops, aidx, amp, cut = propose_dropouts(raw, FS, fraction=.2)
    label = 'unmasked signal, 0.5 Hz high-pass · local prominence max(floor, 12% local span), 40 ms spacing'
    return path, t, y, FS, s, prom, rel, auto_spike, (drops, aidx, amp, cut), dict(
        method='local400', method_label=label, relative_prominence=RELATIVE, minimum_interval_s=MIN_INTERVAL_S), ''


def build_original100(mouse, floor):
    path = find_h5(mouse)
    t, y, bounds = original_signal(path)
    s, prom = original_candidates(t, y, bounds)
    plain = np.concatenate([np.flatnonzero((t >= lo)&(t < hi))[find_peaks(y[(t >= lo)&(t < hi)]-y[(t >= lo)&(t < hi)].mean(), distance=8)[0]]
                            for lo, hi in zip(bounds[:-1], bounds[1:])])
    assert np.array_equal(plain, s), 'floor 0 must equal the original detector exactly'
    rel = np.zeros(len(s))
    drops, aidx, amp, cut = propose_dropouts(y, OFS, fraction=.2, step=10)
    label = ('original method at 100 Hz: clip ±10 MAD, 50 Hz low-pass, resample, 0.1–20 Hz band-pass; '
             'find_peaks 80 ms spacing per phase, no prominence threshold (floor 0 = original)')
    return path, t, y, OFS, s, prom, rel, spike_flag(y, OFS), (drops, aidx, amp, cut), dict(
        method='original100', method_label=label, relative_prominence=0., minimum_interval_s=.08), '_original100'


VIDEO_DIRS = [Path('C:/Users/sjs93/SLEAP_cohort2/videos'), None]  # local copies first; None = the .rec folder on the raw drive
RAW = Path('D:/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc')
# Shared lab Dropbox copy of the same recordings. Pages built with --dropbox live inside it and point
# at each video by a RELATIVE path, so they work on any computer that syncs this Dropbox folder.
DROPBOX = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc')
DROPBOX_REVIEW = DROPBOX/'breath_review'


def video_data(path, dropbox=False):
    """Video location plus each frame's recording time (camera timestamps), for a synced player."""
    name = f'{path.stem}.1.mp4'
    if dropbox:
        assert (DROPBOX/f'{path.stem}.rec'/name).exists(), f'{name} not in the Dropbox folder'
        src = f'../../{path.stem}.rec/{name}'  # relative to breath_review/review_pages/
    else:
        for d in VIDEO_DIRS:
            mp4 = (d/name) if d else RAW/f'{path.stem}.rec'/name
            if mp4.exists(): break
        else:
            mp4 = None
        src = mp4.as_uri() if mp4 else ''
    with h5py.File(path, 'r') as h:
        frame_rec = h['video/recording_time_s'][:]; fps = float(h['video'].attrs['nominal_fps'])
    return dict(video_src=src, video_name=name, video_fps=fps,
                frame_rec_b64=base64.b64encode(frame_rec.astype('<f8').tobytes()).decode())


def page_data(mouse, method='original100', dropbox=False, floor=25.):
    """Everything the review page embeds for one recording (also used by build_practice_review_html.py)."""
    builder = build_original100 if method == 'original100' else build_local400
    path, t, y, fs, s, prom, rel, auto_spike, (drops, aidx, amp, cut), meta, suffix = builder(mouse, floor)
    edges = np.diff(np.r_[False, auto_spike, False].astype(int))
    data = dict(mouse=mouse, h5=path.name, fs=fs, t0=float(t[0]), n=len(y), floor=floor, **meta,
                y_b64=base64.b64encode(y.astype('<f4').tobytes()).decode(),
                cand=s.tolist(), prom=prom.tolist(), rel=rel.tolist(),  # unrounded so threshold ties match Python
                auto_spike=[[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))],  # hint only
                amp_step=int(aidx[1]-aidx[0]), amp_log10=np.round(np.log10(np.maximum(amp, 1)), 3).tolist(),
                amp_cut_log10=np.round(np.log10(np.maximum(.2*cut, 1)), 3).tolist(),
                dropouts=[[int(a), int(b)] for a, b in drops],  # hint only; detection unchanged
                save_hint='breath_review/saved_reviews/ (lab Dropbox, next to the review pages)' if dropbox else 'manual_peak_review/saved_reviews/',
                **video_data(path, dropbox))
    return data, dict(path=path, t=t, y=y, s=s, prom=prom, rel=rel, meta=meta)


def render(data):
    html = (ROOT/'scripts/peak_review_template.html').read_text(encoding='utf-8')
    return html.replace('__DATA__', json.dumps(data).replace('</', '<\\/'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mouse', default='11_3', help='recording name prefix, e.g. 11_3 or 13_2_d_CM_13_1_s_20260729_181321')
    ap.add_argument('--method', choices=['original100', 'local400'], default='original100')
    ap.add_argument('--dropbox', action='store_true', help='write the page into the lab Dropbox (breath_review/) for other reviewers')
    ap.add_argument('--prominence', type=float, help='starting prominence floor in ADC (default 25; 0 = no floor, i.e. the original detector)')
    args = ap.parse_args()
    floor = args.prominence if args.prominence is not None else 25.
    data, x = page_data(args.mouse, args.method, args.dropbox, floor)
    meta, s, prom, rel = x['meta'], x['s'], x['prom'], x['rel']
    folder = (DROPBOX_REVIEW/'review_pages') if args.dropbox else OUT/('review_pages_100Hz' if meta['method'] == 'original100' else 'old_400Hz_pages')
    folder.mkdir(parents=True, exist_ok=True)
    out = folder/f'{args.mouse}.html'; out.write_text(render(data), encoding='utf-8')
    n = (prom >= np.maximum(floor, rel)).sum()
    print(f'{out}\n{meta["method"]}: {len(s)} candidates, {n} pass at floor {floor:g}, {out.stat().st_size/1e6:.1f} MB')
