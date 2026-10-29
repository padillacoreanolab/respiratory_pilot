"""Pick example 5 s pages for breath-review training (narrated video + practice page).

Scores every page of each cohort 2 recording with the same signal and peaks as the review page
(original 100 Hz method, floor 25 ADC) and sorts pages into teaching cases:
  clean     no flags, 2-6.5 breaths/s, at most 2 sub-floor candidates
  missed    >= 3 sub-floor candidates (prominence 8-25 ADC) that may be real small breaths
  dropout   >= 50% covered by the low-amplitude hint (signal dropped out)
  spike     overlaps the large-spike hint, or peaks above 8 robust SD (handling, air push, tubing knock)
  sniffing  >= 7 detected peaks/s with no flags (fast breathing is real)
  social    clean page from the cagemate phase (contact and fast movement)
Writes outputs/aim1_summer2026_ranked_recs/manual_peak_review/training/training_pages.json.

Usage (biopipeline-env): python scripts/select_training_pages.py [--mouse 10_3]
"""
from pathlib import Path
import sys, json, argparse
import numpy as np, h5py

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT))
from build_peak_review_html import page_data
from review_peaks_gui import D as H5DIR

PAGE_S, FLOOR, SUB = 5, 25., 8.
BIG = 8.  # page peak |signal| in robust SDs of the whole recording: a large transient
OUT = ROOT/'outputs/aim1_summer2026_ranked_recs/manual_peak_review/training'
WANT = {'clean': 3, 'missed': 4, 'dropout': 3, 'spike': 3, 'sniffing': 3, 'social': 4}


def score(mouse):
    data, x = page_data(mouse, floor=FLOOR)
    fs, n, s, prom = data['fs'], data['n'], x['s'], x['prom']
    with h5py.File(x['path'], 'r') as h: social0 = float(h['phases/video_boundary_s'][1])
    def cover(ivs, a, b): return sum(max(0, min(b, q)-max(a, p)) for p, q in ivs)/(b-a)
    y = x['y']; sd = 1.4826*np.median(np.abs(y-np.median(y)))
    rows = []
    for i in range(int(np.ceil(n/fs/PAGE_S))):
        a, b = int(i*PAGE_S*fs), min(n, int((i+1)*PAGE_S*fs))
        if b-a < PAGE_S*fs*.9: continue
        m = (s >= a)&(s < b); kept = int((prom[m] >= FLOOR).sum()); sub = int(((prom[m] >= SUB)&(prom[m] < FLOOR)).sum())
        rows.append(dict(page=i, start_s=round(data['t0']+i*PAGE_S, 2), kept=kept, sub=sub, rate=kept/PAGE_S,
                         drop=cover(data['dropouts'], a, b), spike=cover(data['auto_spike'], a, b),
                         big=float(np.nanmax(np.abs(y[a:b]))/sd),
                         social=data['t0']+i*PAGE_S >= social0))
    return data, rows, social0


def classify(r):
    spike = r['spike'] > 0 or r['big'] >= BIG
    flags = r['drop'] > .05 or spike
    out = []
    if r['drop'] >= .5: out.append('dropout')
    if spike and r['drop'] < .5: out.append('spike')
    if not flags and r['sub'] >= 3: out.append('missed')
    if not flags and r['rate'] >= 7: out.append('sniffing')
    if not flags and 2 <= r['rate'] <= 6.5 and r['sub'] <= 2: out.append('social' if r['social'] else 'clean')
    return out


def pick(rows):
    """Up to WANT[c] pages per case, spread over the recording, no page used twice."""
    used, chosen = set(), []
    for case, k in WANT.items():
        pool = [r for r in rows if case in classify(r) and r['page'] not in used]
        if not pool: continue
        idx = np.linspace(0, len(pool)-1, min(k, len(pool))).round().astype(int)
        for j in dict.fromkeys(idx):
            used.add(pool[j]['page']); chosen.append(dict(pool[j], case=case))
    return sorted(chosen, key=lambda r: r['page'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mouse'); args = ap.parse_args()
    mice = [args.mouse] if args.mouse else sorted({p.stem[:4] if not p.stem.startswith('13_2_') else p.stem for p in H5DIR.glob('*.h5')})
    best = None
    for mouse in mice:
        data, rows, social0 = score(mouse)
        chosen = pick(rows)
        cover_ = {c: sum(r['case'] == c for r in chosen) for c in WANT}
        print(mouse, len(rows), 'pages |', cover_, flush=True)
        key = (sum(min(v, 1) for v in cover_.values()), len(chosen))
        if best is None or key > best[0]: best = (key, mouse, data, chosen, social0)
    _, mouse, data, chosen, social0 = best
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(mouse=mouse, h5=data['h5'], page_s=PAGE_S, floor=FLOOR, social_start_s=social0,
                   pages=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in chosen]),
              open(OUT/'training_pages.json', 'w'), indent=1)
    print('chose', mouse, '->', OUT/'training_pages.json')
    for r in chosen: print(f"  page {r['page']+1:4d}  {r['start_s']:7.1f} s  {r['case']:9s} kept {r['kept']:2d} sub {r['sub']:2d} drop {r['drop']:.2f} spike {r['spike']:.2f}")
