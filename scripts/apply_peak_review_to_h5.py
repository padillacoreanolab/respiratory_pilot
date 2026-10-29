"""Write manually reviewed breath peaks into the session H5.

Adds (or replaces) only the group respiration/manual_reviewed_breaths; every
existing dataset is left untouched. Peaks are rebuilt from the review JSON
(detector rerun on the UNMASKED signal at the saved floor, minus removed, plus added,
minus anything inside the reviewer's unusable stretches), so the H5 does not depend
on the CSV. The reviewer's exclusions are the only mask; the automatic spike and
low-amplitude flags are stored separately for reference. A backup copy is made
before the first write.

Each method writes its own group: original100 -> respiration/manual_reviewed_breaths_original100,
local400 -> respiration/manual_reviewed_breaths.

Usage (biopipeline-env):
  python scripts/apply_peak_review_to_h5.py --mouse 11_3                      # original100 review
  python scripts/apply_peak_review_to_h5.py --mouse 11_3 --method local400
  python scripts/apply_peak_review_to_h5.py --mouse 11_3 --review path/to/file.json --dry-run
"""
from pathlib import Path
import sys, json, shutil, argparse
from datetime import datetime
import numpy as np, h5py

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from review_peaks_gui import load_unmasked, find_h5, detect, OUT, FS
from src.rsp.signal_quality import propose_dropouts
from src.rsp.original_method import original_signal, detect_original, FS as OFS
from build_peak_review_html import spike_flag
GROUPS = {'local400': 'respiration/manual_reviewed_breaths', 'original100': 'respiration/manual_reviewed_breaths_original100'}

ap = argparse.ArgumentParser()
ap.add_argument('--mouse', default='11_3'); ap.add_argument('--review', type=Path)
ap.add_argument('--method', choices=list(GROUPS), default='original100', help='which review page the JSON came from')
ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--no-backup', action='store_true')
ap.add_argument('--allow-partial', action='store_true', help='write even if not every 5 s page was marked reviewed')
args = ap.parse_args()
DROPBOX_SAVED = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc/breath_review/saved_reviews')
fname = f"{args.mouse}{'_original100' if args.method == 'original100' else ''}_peak_review.json"
candidates_paths = [p for p in (OUT/'saved_reviews'/fname, DROPBOX_SAVED/fname) if p.exists()]
if args.review: review_path = args.review
elif len(candidates_paths) == 1: review_path = candidates_paths[0]
elif candidates_paths: raise SystemExit(f'{fname} exists both locally and in Dropbox; pass --review with the one to use')
else: raise SystemExit(f'{fname} not found in manual_peak_review/saved_reviews or Dropbox breath_review/saved_reviews')
print('review:', review_path)
review = json.loads(review_path.read_text())
if 'excluded_intervals' not in review:
    raise SystemExit('This review JSON was saved from the older masked page; reopen the rebuilt page, load it, and save again.')

method = review.get('method', 'local400')
if method != args.method: raise SystemExit(f'review JSON is for method {method}; pass --method {method}')
GROUP = GROUPS[method]
if method == 'original100':
    path = find_h5(args.mouse); t, y, bounds = original_signal(path); fs = OFS
    detected = detect_original(t, y, bounds, review['minimum_prominence_adc'])
    auto_spike = spike_flag(y, fs); low_source, low_step = y, 10
    detector_text = ('src/rsp/original_method.py: raw 20 kHz, clip +/-10 MAD, order-4 50 Hz low-pass, resample_poly to 100 Hz, '
                     'NeuroKit 0.1-20 Hz band-pass; find_peaks 80 ms spacing per phase, prominence >= floor (0 = original)')
    grid_text = 'original 100 Hz grid: time_s = (first acquisition tick - recording_tick_origin)/20000 + sample/100'
else:
    path, t, y, auto_spike = load_unmasked(args.mouse); fs = FS
    detected = detect(y, np.zeros(len(y), bool), review['minimum_prominence_adc'])
    with h5py.File(path, 'r') as h: low_source = h['breathmetrics/pressure_preprocessed'][:].astype(float)
    low_step = 40
    detector_text = 'src/rsp/local_peaks.py on unmasked breathmetrics/pressure_preprocessed after a whole-recording 0.5 Hz high-pass'
    grid_text = 'breathmetrics/time_s'
assert review['h5'] == path.name, f"review is for {review['h5']}, not {path.name}"
assert review.get('fs', fs) == fs, f"review sampled at {review.get('fs')} Hz, method uses {fs} Hz"
excluded = np.zeros(len(y), bool)
for a, b in review['excluded_intervals']: excluded[int(a):int(b)] = True
removed = np.array(review['removed_samples'], dtype=int); added = np.array(review['added_samples'], dtype=int)
final = np.union1d(np.setdiff1d(detected, removed), added)
final = final[~excluded[final]]
assert len(final) == review['n_final'], f"rebuilt {len(final)} peaks but review saved {review['n_final']}"

ibi = np.diff(t[final])
# An interval is valid only if no reviewer-excluded sample lies between its two peaks.
excl_before = np.r_[0, np.cumsum(excluded)]
interval_valid = (excl_before[final[1:]] - excl_before[final[:-1]]) == 0
pages = review.get('review_pages', {})
reviewed = len(pages.get('reviewed', [])) / pages['n_pages'] if pages.get('n_pages') else 0.
if reviewed < 1 and not args.dry_run and not args.allow_partial:
    raise SystemExit(f'Only {reviewed:.0%} of pages are marked reviewed; finish the full review or pass --allow-partial.')
print(f'{path.name}: {len(final)} peaks ({len(detected)} detected at {review["minimum_prominence_adc"]:g} ADC, '
      f'{len(removed)} removed, {len(added)} added, {excluded.sum()/fs:.1f} s marked unusable); '
      f'{interval_valid.sum()}/{len(ibi)} intervals valid; {reviewed:.0%} of pages reviewed')
print(f'median rate over valid intervals: {np.median(1/ibi[interval_valid]):.2f} Hz')
if args.dry_run: sys.exit()

auto_low = np.zeros(len(y), bool)
for a, b in propose_dropouts(low_source, fs, fraction=.2, step=low_step)[0]: auto_low[a:b] = True

bdir = OUT/'h5_backups'
if not args.no_backup and not list(bdir.glob(f'{path.stem}.before_manual_review_*.h5')):
    # Keep only the pre-review original; later runs just replace the review group.
    bdir.mkdir(parents=True, exist_ok=True)
    backup = bdir/f'{path.stem}.before_manual_review_{datetime.now():%Y%m%d_%H%M%S}.h5'
    shutil.copy2(path, backup); print(f'backup: {backup}')

with h5py.File(path, 'r+') as h:
    if GROUP in h: del h[GROUP]
    g = h.create_group(GROUP)
    g.attrs.update(dict(
        status='Manually reviewed local-prominence candidates; positive pressure peaks, phase not verified as inhalation onset',
        method=method, detector=detector_text,
        mask_policy='Only reviewer-marked unusable stretches are excluded; automatic flags are stored for reference and never applied',
        minimum_prominence_adc=review['minimum_prominence_adc'], relative_prominence=review['relative_prominence'],
        minimum_interval_s=review['minimum_interval_s'], sampling_rate_hz=fs, sample_grid=grid_text,
        reviewed_at=review['saved'], written_at=datetime.now().isoformat(timespec='seconds'),
        reviewed_fraction=reviewed, review_page_s=pages.get('page_s', 0), excluded_s=float(excluded.sum()/fs),
        review_json=json.dumps(review)))
    g['peak_sample_index'] = final
    g['peak_time_s'] = t[final]; g['peak_time_s'].attrs['units'] = 's'
    g['peak_source'] = np.where(np.isin(final, detected), 'detected', 'added').astype('S8')
    g['interpeak_interval_s'] = ibi; g['interpeak_interval_s'].attrs['units'] = 's'
    g['interval_rate_hz'] = 1/ibi; g['interval_rate_hz'].attrs['units'] = 'Hz'
    g['interval_valid'] = interval_valid
    g['interval_valid'].attrs['meaning'] = 'False when the interval spans a reviewer-marked unusable stretch; exclude from rate analyses'
    g['removed_sample_index'] = removed; g['added_sample_index'] = added
    g['reviewer_excluded'] = excluded
    g['reviewer_excluded'].attrs['meaning'] = 'True where the reviewer marked the signal unusable (the only mask applied)'
    g['auto_spike_flag'] = auto_spike; g['auto_spike_flag'].attrs['meaning'] = 'Reference only: >20 robust SD with 3 s padding; not applied'
    g['auto_low_amplitude_flag'] = auto_low; g['auto_low_amplitude_flag'].attrs['meaning'] = 'Reference only: amplitude <20% of 120 s rolling median for >=1 s; not applied'
print(f'wrote {GROUP} to {path}')
