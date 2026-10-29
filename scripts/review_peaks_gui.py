"""Manual review of local-prominence peak candidates (src/rsp/local_peaks.py).

Preprocessing matches scripts/fix_local_resp_detection.py: 20 robust-SD artifact
mask with 3 s padding, 0.5 Hz high-pass inside each continuous segment.
Never modifies the shared H5 files; edits are saved under manual_peak_review/.

Usage (biopipeline-env):
  python scripts/review_peaks_gui.py --mouse 11_3 --prominence 25
  python scripts/review_peaks_gui.py --mouse 11_3 --prominence 25 --replot-only

Controls:
  left click      toggle remove/restore the nearest detected peak (red x = removed)
  right click     add a peak at the local maximum near the click; right click an
                  added peak (orange) to delete it
  left/right      page backward/forward     up/down  zoom in/out
  u               undo last edit            s        save
  prominence box  type a new ADC floor + Enter to re-detect (edits are kept)
"""
from pathlib import Path
import sys, json, argparse
from datetime import datetime
import numpy as np, h5py, pandas as pd
from scipy import signal, ndimage

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.rsp.local_peaks import detect_local_peaks
D = ROOT/'outputs/aim1_summer2026_ranked_recs'
OUT = D/'manual_peak_review'
FS = 400
SNAP_S = .03  # click tolerance / local-max search radius


def find_h5(prefix):
    matches = sorted(D.glob(prefix+'*.h5'))
    if len(matches) != 1:
        raise SystemExit(f'{prefix!r} matches {len(matches)} recordings; pass a longer prefix: {[p.stem for p in matches]}')
    return matches[0]


def load_clean(mouse):
    path = find_h5(mouse)
    with h5py.File(path, 'r') as h:
        t = h['breathmetrics/time_s'][:]; y = h['breathmetrics/pressure_preprocessed'][:].astype(float)
    scale = 1.4826*np.median(np.abs(y-np.median(y)))
    bad = ndimage.maximum_filter1d((np.abs(y-np.median(y)) > 20*scale).astype('u1'), size=2401) > 0
    if path.stem.startswith('11_2_'): bad |= (t >= 482)&(t < 516)  # known dropout, set by visual review
    clean = np.full_like(y, np.nan)
    edges = np.diff(np.r_[False, ~bad, False].astype(int))
    sos = signal.butter(2, .5, btype='highpass', fs=FS, output='sos')
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b-a > FS: clean[a:b] = signal.sosfiltfilt(sos, y[a:b])
    return path, t, clean, bad


def load_unmasked(mouse):
    """Whole-recording 0.5 Hz high-pass with nothing removed, plus the automatic spike flag as a hint only.

    Used by the human review page: every stretch stays visible and detectable, and only the
    reviewer's own exclusions decide what is unusable.
    """
    path, t, _, auto_spike = load_clean(mouse)
    with h5py.File(path, 'r') as h: y = h['breathmetrics/pressure_preprocessed'][:].astype(float)
    hp = signal.sosfiltfilt(signal.butter(2, .5, btype='highpass', fs=FS, output='sos'), y)
    return path, t, hp, auto_spike


def detect(clean, bad, prominence):
    return detect_local_peaks(clean, FS, bad, minimum_prominence=prominence)[0]


def replot(mouse, t, clean, bad, prominence, starts=(300, 550, 442), compare=15.):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    new, old = detect(clean, bad, prominence), detect(clean, bad, compare)
    fig, axs = plt.subplots(len(starts), 1, figsize=(14, 3.6*len(starts)), layout='constrained')
    summary = []
    for ax, start in zip(np.atleast_1d(axs), starts):
        end = start+1.5; m = (t >= start)&(t < end)
        qn = new[(t[new] >= start)&(t[new] < end)]; qo = old[(t[old] >= start)&(t[old] < end)]
        dropped = np.setdiff1d(qo, qn)
        ax.plot(t[m], clean[m], lw=1, color='#1677b8')
        ax.scatter(t[qn], clean[qn], facecolors='none', edgecolors='green', s=40, label=f'kept at {prominence:g} ADC')
        ax.scatter(t[dropped], clean[dropped], marker='x', color='#B34A16', s=40, label=f'found at {compare:g} ADC only')
        for n, k in enumerate(qn, 1): ax.annotate(str(n), (t[k], clean[k]), xytext=(0, 6), textcoords='offset points', fontsize=8, ha='center')
        ax.set(title=f'{mouse}: {start}-{end} s | {len(qn)} candidates at {prominence:g} ADC ({len(qo)} at {compare:g})',
               ylabel='ADC counts', xlabel='Recording seconds')
        summary.append(dict(start_s=start, stop_s=end, count=len(qn), count_at_compare=len(qo)))
    np.atleast_1d(axs)[0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{mouse} local-prominence candidates: floor {prominence:g} ADC vs {compare:g} ADC (relative 12%, spacing 40 ms) - not manually validated')
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT/f'qc_{mouse}_prominence{prominence:g}_vs_{compare:g}.png'; fig.savefig(png, dpi=130); plt.close(fig)
    print(png); print(f'whole session: {len(new)} peaks at {prominence:g} ADC, {len(old)} at {compare:g} ADC')
    print(pd.DataFrame(summary).to_string(index=False))


class Reviewer:
    def __init__(self, mouse, path, t, clean, bad, prominence, window_s):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, TextBox, Button
        self.plt = plt
        self.mouse, self.path, self.t, self.y, self.bad = mouse, path, t, clean, bad
        self.prominence, self.window = prominence, window_s
        self.json_path = OUT/f'{mouse}_peak_review.json'
        self.removed, self.added, self.history, self.start = set(), set(), [], float(t[0])
        if self.json_path.exists():
            prev = json.loads(self.json_path.read_text())
            self.removed, self.added = set(prev['removed_samples']), set(prev['added_samples'])
            self.start = prev.get('last_view_start_s', self.start)
            print(f'Resumed {self.json_path.name}: {len(self.removed)} removed, {len(self.added)} added')
        self.detected = detect(clean, bad, prominence)

        self.fig = plt.figure(figsize=(15, 6.5))
        self.ax = self.fig.add_axes([.06, .3, .91, .62])
        self.line, = self.ax.plot([], [], lw=1, color='#1677b8')
        self.kept = self.ax.scatter([], [], facecolors='none', edgecolors='green', s=50, zorder=3)
        self.rem = self.ax.scatter([], [], marker='x', color='red', s=60, zorder=3)
        self.add = self.ax.scatter([], [], marker='o', color='orange', s=50, zorder=3)
        self.spans = []
        self.ax.set(xlabel='Recording seconds', ylabel='ADC counts (drift-reduced)')
        self.slider = Slider(self.fig.add_axes([.06, .17, .78, .03]), 'Start (s)', float(t[0]), float(t[-1]-window_s), valinit=self.start)
        self.slider.on_changed(self.goto)
        self.box = TextBox(self.fig.add_axes([.17, .07, .08, .05]), 'Prominence floor (ADC) ', initial=f'{prominence:g}')
        self.box.on_submit(self.redetect)
        self.save_btn = Button(self.fig.add_axes([.30, .07, .08, .05]), 'Save (s)'); self.save_btn.on_clicked(lambda e: self.save())
        self.undo_btn = Button(self.fig.add_axes([.40, .07, .08, .05]), 'Undo (u)'); self.undo_btn.on_clicked(lambda e: self.undo())
        self.status = self.fig.text(.50, .085, '', fontsize=9)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('key_press_event', self.key)
        self.draw()
        plt.show()

    def final(self):
        keep = set(self.detected.tolist())-self.removed
        return np.array(sorted(keep | self.added), dtype=int)

    def draw(self):
        t, y = self.t, self.y
        a, b = np.searchsorted(t, [self.start, self.start+self.window])
        self.line.set_data(t[a:b], y[a:b])
        det = self.detected[(self.detected >= a)&(self.detected < b)]
        rem = np.array([k for k in det if k in self.removed], dtype=int)
        kept = np.array([k for k in det if k not in self.removed], dtype=int)
        add = np.array([k for k in self.added if a <= k < b], dtype=int)
        for sc, idx in [(self.kept, kept), (self.rem, rem), (self.add, add)]:
            sc.set_offsets(np.c_[t[idx], y[idx]] if len(idx) else np.empty((0, 2)))
        for s in self.spans: s.remove()
        edges = np.diff(np.r_[False, self.bad[a:b], False].astype(int))
        self.spans = [self.ax.axvspan(t[a+i], t[min(a+j, len(t)-1)], color='red', alpha=.15)
                      for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        self.ax.set_xlim(self.start, self.start+self.window)
        seg = y[a:b][np.isfinite(y[a:b])]
        if len(seg): pad = .1*np.ptp(seg)+1; self.ax.set_ylim(seg.min()-pad, seg.max()+pad)
        self.ax.set_title(f'{self.mouse} | floor {self.prominence:g} ADC | in view: {len(kept)} kept, {len(rem)} removed, {len(add)} added'
                          f' | session total: {len(self.final())} | red shading = masked signal loss')
        self.status.set_text(f'{len(self.removed)} removed, {len(self.added)} added (all edits) | left click: toggle remove, right click: add')
        self.fig.canvas.draw_idle()

    def goto(self, val):
        self.start = float(np.clip(val, self.t[0], self.t[-1]-self.window)); self.draw()

    def page(self, frac):
        self.slider.set_val(np.clip(self.start+frac*self.window, self.slider.valmin, self.slider.valmax))

    def nearest(self, idx, x):
        if not len(idx): return None
        k = idx[np.argmin(np.abs(self.t[idx]-x))]
        return int(k) if abs(self.t[k]-x) <= SNAP_S*self.window/2 else None

    def click(self, e):
        if e.inaxes is not self.ax or e.xdata is None or getattr(self.fig.canvas.toolbar, 'mode', ''): return  # ignore clicks while zoom/pan tool is active
        if e.button == 1:
            k = self.nearest(self.detected, e.xdata)
            if k is None: return
            self.removed.symmetric_difference_update({k}); self.history.append(('removed', k))
        elif e.button == 3:
            k = self.nearest(np.array(sorted(self.added), dtype=int), e.xdata)
            if k is None:
                c = np.searchsorted(self.t, e.xdata); r = int(SNAP_S*FS)
                lo, hi = max(0, c-r), min(len(self.y), c+r)
                if not np.isfinite(self.y[lo:hi]).any(): return
                k = lo+int(np.nanargmax(self.y[lo:hi]))
                if k in self.detected: return  # already a candidate; left click restores it
            self.added.symmetric_difference_update({k}); self.history.append(('added', k))
        self.draw()

    def undo(self):
        if not self.history: return
        which, k = self.history.pop()
        getattr(self, which).symmetric_difference_update({k}); self.draw()

    def key(self, e):
        {'right': lambda: self.page(.8), 'left': lambda: self.page(-.8), 'u': self.undo, 's': self.save,
         'up': lambda: self.zoom(.5), 'down': lambda: self.zoom(2)}.get(e.key, lambda: None)()

    def zoom(self, f):
        self.window = float(np.clip(self.window*f, .5, 60))
        self.slider.valmax = float(self.t[-1]-self.window); self.goto(self.start)

    def redetect(self, text):
        try: self.prominence = float(text)
        except ValueError: return
        self.detected = detect(self.y, self.bad, self.prominence); self.draw()

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        p = self.final(); det = set(self.detected.tolist())
        pd.DataFrame(dict(sample=p, time_s=self.t[p], source=['added' if k not in det else 'detected' for k in p])
                     ).to_csv(OUT/f'{self.mouse}_reviewed_peaks.csv', index=False)
        self.json_path.write_text(json.dumps(dict(
            h5=self.path.name, saved=datetime.now().isoformat(timespec='seconds'),
            detector='src/rsp/local_peaks.py', minimum_prominence_adc=self.prominence, relative_prominence=.12,
            minimum_interval_s=.04, fs=FS, n_detected=len(self.detected), n_final=len(p),
            removed_samples=sorted(int(k) for k in self.removed), added_samples=sorted(int(k) for k in self.added),
            last_view_start_s=self.start), indent=2))
        self.status.set_text(f'Saved {len(p)} peaks -> {OUT.relative_to(ROOT)}'); self.fig.canvas.draw_idle()
        print(f'Saved {len(p)} peaks to {OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mouse', default='11_3'); ap.add_argument('--prominence', type=float, default=25.)
    ap.add_argument('--window', type=float, default=5.); ap.add_argument('--replot-only', action='store_true')
    args = ap.parse_args()
    path, t, clean, bad = load_clean(args.mouse)
    if args.replot_only: replot(args.mouse, t, clean, bad, args.prominence)
    else: Reviewer(args.mouse, path, t, clean, bad, args.prominence, args.window)
