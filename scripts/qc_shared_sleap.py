"""Tracking-quality summary for every SLEAP pose set stored in the shared H5s (D:/R34_shared_h5).

Per recording: both mice found (% frames), body points found (% of nodes), median point score,
jumps per minute (a track's centroid moving > 10% of the arena width between frames: swaps or bad frames),
and frames where the two tracks sit on top of each other (< 3% of the arena width: two skeletons on one mouse).
Output: outputs/r34_shared/sleap_qc.csv.  Usage (biopipeline-env): python scripts/qc_shared_sleap.py
"""
from pathlib import Path
import numpy as np, h5py, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(Path('D:/R34_shared_h5').glob('*/*.h5')):
    with h5py.File(f, 'r') as h:
        if 'behavior/sleap' not in h: continue
        s = h['behavior/sleap']; xy = s['pose_xy'][:]; sc = s['point_scores'][:] if 'point_scores' in s else None
        fps = h['video'].attrs['fps']; aligned = bool(s.attrs['frames_match_video'])
    found = np.isfinite(xy[..., 0])                                   # frames x tracks x nodes
    with np.errstate(all='ignore'):
        c = np.nanmean(xy, axis=2)                                    # frames x tracks x 2
        width = np.nanpercentile(c[..., 0], 99) - np.nanpercentile(c[..., 0], 1)
        step = np.hypot(*np.diff(c, axis=0).transpose(2, 0, 1))       # frames-1 x tracks
        sep = np.hypot(*(c[:, 0] - c[:, 1]).T)
    minutes = len(xy) / fps / 60
    rows.append(dict(cohort=f.parent.name, session=f.stem, frames=len(xy), aligned=aligned,
                     both_found_pct=round(100 * found.any(2).all(1).mean(), 1),
                     nodes_found_pct=round(100 * found[found.any(2)].mean(), 1),
                     median_score=round(float(np.nanmedian(sc[found])), 2) if sc is not None else np.nan,
                     jumps_per_min=round(float((step > .1 * width).sum() / minutes), 1),
                     overlap_pct=round(100 * float(np.nanmean(sep < .03 * width)), 1)))
df = pd.DataFrame(rows); df.to_csv(ROOT/'outputs/r34_shared/sleap_qc.csv', index=False)
print(df.to_string(index=False))
