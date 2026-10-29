"""List every R34 paper recording with its raw folder, BORIS export(s), SLEAP file and phase boundaries.

Output: outputs/r34_shared/manifest.json, the input of scripts/convert_shared_h5.py.
Files are matched by the recording timestamp (YYYYMMDD_HHMMSS), which is unique per recording,
because file names differ between folders (e.g. pilot1 BORIS '1_1_2_i_cm_1_2_d_...' vs .rec '1_1_d_cm_1_2_i_...').
Read-only. Needs D: (easystore) and E: (Black Widow).

Usage (biopipeline-env): python scripts/build_shared_manifest.py
"""
from pathlib import Path
import json, re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/r34_shared'
R34 = Path('D:/R34_working')
COHORT2 = Path('D:/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc')
PILOT1 = Path('D:/BLA_ChR_resp/BLA_resp_ChR_cm_hc')
SLEAP_VALENCE = Path('E:/Aim1/AIM1/Day1_new/SLEAP Tracked')
ISO = [(Path('E:/Aim1/AIM1/cagemate_1d_iso'), '1-day isolation'), (Path('D:/Aim1/AIM1'), '10-day isolation')]
STAMP = re.compile(r'(20\d{6}_\d{6})')
# cohort 2 ranks and partners from the recording notes (same source as scripts/aim1_summer2026_boris.py)
RANKS = {'13.2': 1, '13.1': 4, '13.3': 3, '14.3': 4, '14.1': 1, '10.2': 1, '10.1': 4, '10.3': 3, '11.2': 2, '11.3': 3, '11.4': 1}


def stamp(p):
    m = STAMP.search(Path(p).name)
    return m.group(1) if m else None


def subject(stem):
    m = re.match(r'(?:BLRI|RI1|RI2|BL|CM|test)_s(\d)_(\d+)', stem) or re.match(r'(\d+)_(\d)_', stem)
    return f'{m.group(1)}.{m.group(2)}' if m else None


def by_stamp(paths):
    out = {}
    for p in paths:
        if stamp(p): out.setdefault(stamp(p), []).append(str(p).replace('\\', '/'))
    return out


def row(rec_dir, experiment, cohort, phase, boris=(), sleap=None, bounds=None):
    stem = rec_dir.name[:-4]
    sid = subject(stem)
    return dict(session_id=stem, experiment=experiment, cohort=cohort, phase=phase, subject_id=sid,
                subject_rank=RANKS.get(sid) if cohort.startswith('Cagemate B') else None,
                rec_dir=str(rec_dir).replace('\\', '/'), boris=sorted(boris), sleap=sleap, video_boundaries=bounds)


rows = []
# Valence: one session folder per recording, BORIS inside it; SLEAP on E:
sleap_v = by_stamp(SLEAP_VALENCE.glob('*.h5'))
chosen = {json.loads(Path(R34/'selected_boris.json').read_text())[i]['working_path'] for i in range(2)}
for d in sorted((R34/'valence').iterdir()):
    rec = next(d.glob('*.rec'), None)
    if rec is None: continue
    phase = d.name.split('_')[0]
    boris = [str(p).replace('\\', '/') for p in (d/'boris').glob('*.csv')] if (d/'boris').exists() else []
    if len(boris) > 1:  # keep the user-confirmed file when there are several
        keep = [b for b in boris if any(b.endswith(c) for c in chosen)]
        boris = keep or boris
    sl = sleap_v.get(stamp(d), [None])[0]
    rows.append(row(rec, 'Valence', 'Valence cohort 1 (Jun 2025)', phase, boris, sl))
# Cagemate A: baseline and cagemate folders in R34_working/ranked/cohort1
CAGEMATE_A_E = Path('E:/Aim1/AIM1/Cagemate_bl_no_ISO')
for d in sorted((R34/'ranked/cohort1').iterdir()):
    rec = next(d.glob('*.rec'), None)
    if rec is None: continue
    if not any(rec.rglob('*analog_ECU_Ain1.dat')) and (CAGEMATE_A_E/rec.name).exists():
        rec = CAGEMATE_A_E/rec.name  # R34_working baseline copies lack the analog export; the E: originals have it
    boris = [str(p).replace('\\', '/') for p in (d/'boris').glob('*.csv')] if (d/'boris').exists() else []
    rows.append(row(rec, 'Cagemate', 'Cagemate A (Jun 2025)', 'Baseline' if d.name.startswith('BL') else 'Cagemate', boris))
# Cagemate B: baseline and cagemate inside one recording; reviewed video boundaries
bounds = {b['session_id']: b for b in json.loads((ROOT/'manuscript/aim1/summer2026_video_boundaries.json').read_text())['sessions']}
boris_b = by_stamp((COHORT2/'BORIS_csv').glob('*.csv'))
for rec in sorted(COHORT2.glob('*.rec')):
    b = bounds.get(rec.name[:-4])
    rows.append(row(rec, 'Cagemate', 'Cagemate B (Jul 2026)', 'Baseline + cagemate', boris_b.get(stamp(rec), []), None,
                    [b['baseline_start_s'], b['social_start_s'], b['social_end_s']] if b else None))
# Cagemate C: pilot1, no baseline
boris_c = by_stamp((PILOT1/'BLA_resp_ChR_cm_boris').glob('*.csv'))
for rec in sorted(PILOT1.glob('*.rec')):
    rows.append(row(rec, 'Cagemate', 'Cagemate C (Jan 2026)', 'Cagemate', boris_c.get(stamp(rec), [])))
# Isolation: baseline and cagemate are separate recordings. BORIS (cagemate recordings only) is matched by
# timestamp when the export has one, otherwise by the name before it (e.g. 'CM_s3_5_d3_6_10d_iso').
CM_DROPBOX = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2025/ECG_cohort1/Aim1/AIM1/Day1_new/CM interactions')
ISO_BORIS = sorted((CM_DROPBOX/'cm_boris').glob('*d iso/*.csv'))
sleap_iso = by_stamp((CM_DROPBOX/'1_day_isolation/SLEAP Analysis').glob('*.analysis.h5'))
sleap_iso.update(by_stamp((CM_DROPBOX/'10_day_isolation/SLEAP Analysis').glob('*.analysis.h5')))
# Dropbox holds the full isolation set; the D:/E: folders miss five 10-day recordings (BL 3.5/4.7/4.8, CM 4.7/4.8)
# (copied to D:/R34_working/isolation/10d_iso first: the Dropbox paths exceed the Windows 260-character limit)
ISO += [(R34/'isolation/10d_iso', '10-day isolation'), (CM_DROPBOX/'10_day_isolation', '10-day isolation')]
# Misnamed .rec files (Notion R34 RI1/RI2 Boris, row "1.1 CM 10d iso": file named CM_s1_2_d1_1_10d_iso is mouse 1.1)
SUBJECT_FIX = {'CM_s1_2_d1_1_10d_iso_20250704_174528': '1.1'}
seen = set()
for folder, label in ISO:
    for rec in sorted(folder.glob('*_iso*.rec')) + sorted(folder.glob('*_ios_*.rec')):
        stem = rec.name[:-4]
        if stem in seen: continue
        seen.add(stem)
        kind = 'baseline' if rec.name.startswith('BL') else 'cagemate'
        prefix = re.split(r'_\d{8}_\d{6}', stem)[0]
        boris = [str(p).replace('\\', '/') for p in ISO_BORIS if kind == 'cagemate' and
                 (stamp(p) == stamp(rec) if stamp(p) else p.stem.startswith(prefix + '_'))]
        rows.append(row(rec, 'Isolation', 'Isolation (cagemate A mice)', f'{label}, {kind}', boris, sleap_iso.get(stamp(rec), [None])[0]))
        if stem in SUBJECT_FIX: rows[-1]['subject_id'] = SUBJECT_FIX[stem]

OUT.mkdir(parents=True, exist_ok=True)
(OUT/'manifest.json').write_text(json.dumps(rows, indent=1))
print(len(rows), 'recordings ->', OUT/'manifest.json')
for c in sorted({r['cohort'] for r in rows}):
    rr = [r for r in rows if r['cohort'] == c]
    print(f"  {c:30s} {len(rr):3d} recordings, BORIS {sum(bool(r['boris']) for r in rr):3d}, SLEAP {sum(bool(r['sleap']) for r in rr):3d}, multi-BORIS {sum(len(r['boris']) > 1 for r in rr)}")
