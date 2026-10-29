"""Convert R34 paper recordings into one shared H5 format: raw + preprocessed respiration, detected and
reviewed breaths, video frame times, BORIS events aligned to the respiration clock, and SLEAP poses.

Input: outputs/r34_shared/manifest.json (scripts/build_shared_manifest.py). Sources are only read.
Output: D:/R34_shared_h5/<cohort>/<session_id>.h5 (schema 'r34-shared 2.0'; layout in docs/r34_shared_h5_format.txt).

Preprocessing is the lab's original method (src/rsp/original_method.py, docs/respiration_pipeline.txt),
applied on the raw sample grid; every 100 Hz sample is timed by the Trodes tick of its source sample, so
repeated ticks in wireless 'merged' recordings do not shift time. On gap-free recordings the result equals
original_signal()/original_candidates() exactly (checked with --check).

BORIS events are aligned only when the coder's media file has the same length as the original video
(within 2 frames + 0.1 s): media seconds x video fps -> frame -> that frame's Trodes time. Otherwise the
events are stored with alignment_status 'not aligned: media length differs' and NaN recording times.

Usage (biopipeline-env):
  python scripts/convert_shared_h5.py --only 11_2_s_CM            # one session (substring of session_id)
  python scripts/convert_shared_h5.py --cohort "Cagemate B" --check
  python scripts/convert_shared_h5.py                              # everything; existing outputs are skipped
"""
from pathlib import Path
import sys, json, re, argparse, hashlib, datetime, traceback
import numpy as np, h5py, cv2, scipy, neurokit2 as nk, pandas as pd
from scipy.signal import butter, filtfilt, resample_poly, find_peaks

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path('D:/R34_shared_h5')
STR = h5py.string_dtype('utf-8')
FS_OUT, DISTANCE = 100, 8
CITATIONS = (
    'Intranasal cannula pressure recording in freely moving mice: Wesson DW, Donahou TN, Johnson MO, Wachowiak M (2008) '
    'Sniffing behavior of mice during performance in odor-guided tasks. Chem Senses 33(7):581-596. doi:10.1093/chemse/bjn029. '
    'Intranasal pressure in freely behaving mice: Janke E et al. (2022) Machine learning-based clustering and classification of '
    'mouse behaviors via respiratory patterns. iScience 25(12):105625. doi:10.1016/j.isci.2022.105625. '
    'Filter cutoffs, 100 Hz resampling and the 80 ms peak spacing are lab choices (Padilla-Coreano lab), not taken from these papers.')
PREPROCESSING = ('1 subtract median; 2 clip at +/-10 MAD; 3 Butterworth order-4 50 Hz low-pass, zero-phase (filtfilt); '
                 '4 resample_poly to 100 Hz; 5 Butterworth order-2 0.1-20 Hz band-pass, zero-phase (NeuroKit2 signal_filter = sosfiltfilt)')
DETECTION = 'scipy.signal.find_peaks(y - mean(y), distance=8) per phase: 80 ms minimum spacing, no height or prominence threshold'


FRAME_METHOD = ('frame_time_s = straight-line fit of raw frame time vs frame index, separately for each unbroken camera run '
                '(as in src/file_conversion/ekg_resp_to_h5.py); removes USB arrival jitter. frame_time_raw_s = raw .videoTimeStamps. '
                'Checked against the headstage accelerometer (scripts/check_video_sync_imu.py): video ~30 ms late, no drift.')


# Empty BORIS exports that the coder confirmed have no events (recording timestamp -> source of the confirmation).
VERIFIED_NO_EVENTS = {
    '20250623_174348': 'Notion R34 RI1/RI2 Boris, row "3.6 CM Baseline" (Taylor, Vijay A.): "No interactions throughout the '
                       'whole video verified this and no social interaction for the full 10 mins"',
}


def fit_frames(ft):
    """Fitted frame times, number of camera runs, and whether raw timestamps ever run backwards (then all NaN)."""
    d = np.diff(ft)
    if (d < 0).any(): return np.full_like(ft, np.nan), 0, True
    breaks = np.flatnonzero(d > 5*np.median(d)) + 1; out = ft.copy()
    for seg in np.split(np.arange(len(ft)), breaks):
        if len(seg) >= 2: out[seg] = np.polyval(np.polyfit(seg, ft[seg], 1), seg)
    return out, len(breaks)+1, False


def trodes(p, dtype):
    """Trodes .dat/.videoTimeStamps: text header ending '<End settings>', then a binary array."""
    with open(p, 'rb') as f:
        head = []
        for _ in range(200):
            line = f.readline(); head.append(line)
            if b'<End settings>' in line: break
        else:
            raise ValueError(f'{p}: no Trodes header terminator')
        return np.fromfile(f, dtype=dtype), b''.join(head).decode(errors='replace')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def ds(g, name, a, units=None, **attrs):
    a = np.asarray(a)
    x = g.create_dataset(name, data=a, compression='gzip', compression_opts=4, shuffle=True) if a.ndim and a.size else g.create_dataset(name, data=a)
    if units: x.attrs['units'] = units
    x.attrs.update(attrs)
    return x


def text(g, name, values):
    g.create_dataset(name, data=np.array([str(v) for v in values], dtype=object), dtype=STR)


def preprocess(raw, fs):
    """Original method on the sample grid (identical to src/rsp/original_method.original_signal)."""
    x = raw.astype(float) - np.median(raw)
    mad = np.median(np.abs(x))
    if mad > 0: x = np.clip(x, -10*mad, 10*mad)
    b, a = butter(4, 50/(fs/2), btype='low')
    x = resample_poly(filtfilt(b, a, x), FS_OUT, int(fs))
    return np.asarray(nk.signal_filter(x, lowcut=.1, highcut=20, sampling_rate=FS_OUT, order=2), float)


def detect(y, t, bounds):
    idx, prom = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = np.flatnonzero((t >= lo)&(t < hi))
        if not len(m): continue
        ix, p = find_peaks(y[m]-y[m].mean(), distance=DISTANCE, prominence=0)
        idx.append(m[ix]); prom.append(p['prominences'])
    return np.concatenate(idx).astype(int), np.concatenate(prom)


def read_boris(path):
    """BORIS aggregated-events export (both the older and newer column sets)."""
    df = pd.read_csv(path)
    if not len(df): raise ValueError(f'{Path(path).name}: BORIS export has no events')
    col = lambda *names: next((c for c in names if c in df.columns), None)
    need = {'start': col('Start (s)'), 'stop': col('Stop (s)'), 'behavior': col('Behavior'), 'subject': col('Subject')}
    if None in need.values(): raise ValueError(f'{Path(path).name}: not a BORIS aggregated-events export')
    first = lambda c: (pd.to_numeric(df[c], errors='coerce').dropna().iloc[0] if c and pd.to_numeric(df[c], errors='coerce').notna().any() else np.nan)
    meta = dict(media_duration_s=float(first(col('Media duration (s)', 'Total length'))),
                coder_fps=float(first(col('FPS (frame/s)', 'FPS'))),
                time_offset_s=float(first(col('Time offset (s)'))) if col('Time offset (s)') else 0.,
                media_file=str(df[col('Media file name', 'Media file', 'Source')].iloc[0]) if col('Media file name', 'Media file', 'Source') else '')
    typ = df[col('Behavior type')] if col('Behavior type') else pd.Series(['STATE']*len(df))
    return df, need, typ, meta


def convert(r, check=False):
    rec = Path(r['rec_dir']); sid = r['session_id']
    ecu = sorted(rec.rglob('*analog_ECU_Ain1.dat'))
    if len(ecu) != 1: raise ValueError(f'expected one ECU_Ain1.dat under {rec.name}, found {len(ecu)} (needs a Trodes analog export)')
    ts = sorted(ecu[0].parent.glob('*timestamps.dat'))
    if len(ts) != 1: raise ValueError(f'expected one timestamps.dat next to {ecu[0].name}, found {len(ts)}')
    vts = sorted(rec.glob('*.videoTimeStamps')); mp4 = sorted(rec.glob('*.1.mp4'))
    raw, ah = trodes(ecu[0], '<i2'); ticks, th = trodes(ts[0], '<u4')
    fs = float(re.search(r'[Cc]lock ?rate:\s*(\d+)', th)[1])
    if len(raw) != len(ticks): raise ValueError(f'{len(raw)} samples vs {len(ticks)} ticks')
    dt = np.diff(ticks.astype('i8'))
    if (dt < 0).any(): raise ValueError('Trodes clock runs backwards; needs repair')
    origin = int(ticks[0])
    y = preprocess(raw, fs)
    src = np.minimum(np.arange(len(y))*int(round(fs/FS_OUT)), len(ticks)-1)
    t = (ticks[src].astype('f8')-origin)/fs
    # video: each frame's Trodes time on the same clock
    frame_t = frame_raw = fps = nframes = None; video_ok = False
    if vts and mp4:
        vt, vh = trodes(vts[0], '<u4'); frame_raw = (vt.astype('f8')-origin)/fs
        frame_t, n_runs, backwards = fit_frames(frame_raw)
        cap = cv2.VideoCapture(str(mp4[0])); fps = cap.get(cv2.CAP_PROP_FPS); nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        video_ok = bool(fps > 0 and nframes == len(vt) and not backwards)
    to_rec = lambda s: np.interp(np.asarray(s, float)*fps, np.arange(len(frame_t)+1), np.r_[frame_t, frame_t[-1]+1/fps]) if video_ok else np.full(np.shape(s), np.nan)
    # phases (recording seconds)
    if r['video_boundaries'] and video_ok:
        vb = r['video_boundaries']; bounds = [t[0], float(to_rec(vb[1])), t[-1]+.01]; labels = ['baseline', 'cagemate']
        bsrc = 'reviewed video boundaries (whole seconds) mapped through video frame times'
    else:
        bounds = [t[0], t[-1]+.01]; labels = [r['phase']]; bsrc = 'whole recording is one phase (separate recording)'
    peaks, prom = detect(y, t, bounds)
    if check:
        from src.rsp.original_method import original_candidates
        assert (dt == 1).all(), 'check needs a gap-free clock'
        t0 = np.arange(len(y))/FS_OUT
        oi, op = original_candidates(t0, y, np.array(bounds)-t[0])
        assert np.array_equal(oi, peaks) and np.allclose(op, prom), 'peaks differ from original_candidates'
    dest = OUT/r['cohort'].split(' (')[0].replace(' ', '_')/f'{sid}.h5'; dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.partial.h5')
    with h5py.File(tmp, 'w') as h:
        h.attrs.update(schema='r34-shared 2.0', session_id=sid, experiment=r['experiment'], cohort=r['cohort'], phase=r['phase'],
                       subject_id=r['subject_id'] or '', subject_rank=r['subject_rank'] if r['subject_rank'] is not None else -1,
                       created=datetime.datetime.now().isoformat(timespec='seconds'), lab='Padilla-Coreano lab, University of Florida',
                       format_doc='docs/r34_shared_h5_format.txt', citation=CITATIONS)
        g = h.create_group('respiration'); g.attrs.update(sensor='intranasal cannula, pressure transducer; Trodes analog ECU_Ain1', clock_rate_hz=fs, recording_tick_origin=origin)
        rg = g.create_group('raw'); ds(rg, 'pressure_adc', raw, 'ADC counts (uncalibrated)'); ds(rg, 'acquisition_ticks', ticks, 'Trodes ticks')
        rg.attrs.update(time_formula='(acquisition_ticks - recording_tick_origin) / clock_rate_hz', repeated_ticks=int((dt == 0).sum()), clock_gaps=int((dt > 1).sum()))
        pg = g.create_group('preprocessed'); ds(pg, 'pressure', y.astype('f4'), 'ADC counts (filtered)'); ds(pg, 'time_s', t, 's (recording clock)'); ds(pg, 'source_sample_index', src)
        pg.attrs.update(sampling_rate_hz=FS_OUT, method=PREPROCESSING, code='src/rsp/original_method.py', citation=CITATIONS)
        bg = g.create_group('breaths'); ds(bg, 'detected_index', peaks); ds(bg, 'detected_time_s', t[peaks], 's'); ds(bg, 'detected_prominence', prom.astype('f4'), 'ADC counts')
        bg.attrs.update(method=DETECTION, review_status='not reviewed',
                        note='Final breaths come from human review (scripts/apply_peak_review_to_h5.py fills breaths/reviewed). Detected peaks are candidates.')
        rv = bg.create_group('reviewed'); rv.attrs.update(status='not reviewed', fill_with='scripts/apply_peak_review_to_h5.py')
        ph = h.create_group('phases'); ds(ph, 'boundaries_s', bounds, 's (recording clock)'); text(ph, 'labels', labels); ph.attrs['source'] = bsrc
        if frame_t is not None:
            vg = h.create_group('video'); ds(vg, 'frame_time_s', frame_t, 's (recording clock)')
            ds(vg, 'frame_time_raw_s', frame_raw, 's (recording clock)'); ds(vg, 'acquisition_ticks', vt, 'Trodes ticks')
            vg.attrs.update(filename=mp4[0].name, fps=fps, frame_count=nframes, frame_count_matches_timestamps=bool(nframes == len(vt)),
                            frame_timestamps_monotonic=not backwards, frame_origin=0, camera_runs=n_runs,
                            frame_time_method=FRAME_METHOD)
        beh = h.create_group('behavior')
        for k, path in enumerate(r['boris']):
            try:
                df, c, typ, meta = read_boris(path)
            except ValueError as e:
                bb = beh.create_group('boris' if len(r['boris']) == 1 else f'boris_{k}')
                verified = next((v for s, v in VERIFIED_NO_EVENTS.items() if s in Path(path).name), None)
                if verified:   # coder confirmed zero events: a real zero, not missing data
                    for name in ('subject', 'behavior', 'behavior_type'): text(bb, name, [])
                    for name in ('start_media_s', 'stop_media_s', 'start_s', 'stop_s'): ds(bb, name, np.zeros(0), 's')
                    bb.attrs.update(source_file=Path(path).name, alignment_status='aligned (verified: no events)', verified_note=verified)
                else:
                    bb.attrs.update(source_file=Path(path).name, alignment_status=f'not usable: {e}')
                continue
            vlen = nframes/fps if video_ok else np.nan
            ok = video_ok and np.isfinite(meta['media_duration_s']) and abs(meta['media_duration_s']-vlen) <= 2/fps+.1 and not meta['time_offset_s']
            status = ('aligned' if ok else 'not aligned: no video timestamps' if not video_ok else
                      f"not aligned: coder media {meta['media_duration_s']:.2f} s vs video {vlen:.2f} s" if abs(meta['media_duration_s']-vlen) > 2/fps+.1 else
                      f"not aligned: BORIS time offset {meta['time_offset_s']}")
            start, stop = pd.to_numeric(df[c['start']], errors='coerce').to_numpy(), pd.to_numeric(df[c['stop']], errors='coerce').to_numpy()
            bb = beh.create_group('boris' if len(r['boris']) == 1 else f'boris_{k}')
            text(bb, 'subject', df[c['subject']]); text(bb, 'behavior', df[c['behavior']]); text(bb, 'behavior_type', typ)
            ds(bb, 'start_media_s', start, 's (coder media file)'); ds(bb, 'stop_media_s', stop, 's (coder media file)')
            ds(bb, 'start_s', to_rec(start) if ok else np.full(len(df), np.nan), 's (recording clock)')
            ds(bb, 'stop_s', to_rec(stop) if ok else np.full(len(df), np.nan), 's (recording clock)')
            bb.attrs.update(source_file=Path(path).name, alignment_status=status, coder_media_file=meta['media_file'],
                            coder_media_duration_s=meta['media_duration_s'], coder_fps=meta['coder_fps'],
                            method='recording time = frame time of (media seconds x video fps), linear between frames; '
                                   'valid only if the coder media file is the original video (length checked)')
        if r['sleap']:
            with h5py.File(r['sleap'], 'r') as s:
                if 'tracks' in s:
                    tr = s['tracks'][:].transpose(3, 0, 2, 1).astype('f4')  # -> frames x tracks x nodes x (x, y)
                    sg = beh.create_group('sleap'); ds(sg, 'pose_xy', tr, 'pixels')
                    if 'point_scores' in s: ds(sg, 'point_scores', s['point_scores'][:].transpose(2, 0, 1).astype('f4'))
                    text(sg, 'track_names', [n.decode() if isinstance(n, bytes) else n for n in s['track_names'][:]])
                    text(sg, 'node_names', [n.decode() if isinstance(n, bytes) else n for n in s['node_names'][:]])
                    if frame_t is not None and tr.shape[0] == len(frame_t): ds(sg, 'frame_time_s', frame_t, 's (recording clock)')
                    sg.attrs.update(source_file=Path(r['sleap']).name, frames_match_video=bool(frame_t is not None and tr.shape[0] == len(frame_t)),
                                    identity_status='as exported; check ID correction status before use')
        pv = h.create_group('provenance')
        pv.create_dataset('sources_json', dtype=STR, data=json.dumps([dict(path=str(p), bytes=Path(p).stat().st_size, sha256=sha(p)) for p in [ecu[0], ts[0]] + vts]
                                                                   + [dict(path=p) for p in r['boris'] + ([r['sleap']] if r['sleap'] else [])]))
        pv.create_dataset('trodes_headers_json', dtype=STR, data=json.dumps(dict(analog=ah, timestamps=th)))
        pv.attrs.update(builder='scripts/convert_shared_h5.py', builder_sha256=sha(__file__),
                        software=json.dumps(dict(numpy=np.__version__, scipy=scipy.__version__, neurokit2=nk.__version__, h5py=h5py.__version__, opencv=cv2.__version__)))
    with h5py.File(tmp, 'r') as h:  # verify before publishing
        assert np.array_equal(h['respiration/raw/pressure_adc'][:], raw) and np.array_equal(h['respiration/raw/acquisition_ticks'][:], ticks)
    tmp.replace(dest)
    return dict(session_id=sid, file=str(dest), samples=len(raw), duration_s=round(float(t[-1]), 1), breaths=len(peaks),
                repeated_ticks=int((dt == 0).sum()), clock_gaps=int((dt > 1).sum()), video_ok=video_ok,
                boris=[h5py.File(dest, 'r')['behavior'][k].attrs['alignment_status'] for k in h5py.File(dest, 'r')['behavior'] if k.startswith('boris')],
                sleap=bool(r['sleap']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only'); ap.add_argument('--cohort'); ap.add_argument('--check', action='store_true'); ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    rows = json.loads((ROOT/'outputs/r34_shared/manifest.json').read_text())
    rows = [r for r in rows if (not a.only or a.only in r['session_id']) and (not a.cohort or r['cohort'].startswith(a.cohort))]
    report_path = ROOT/'outputs/r34_shared/conversion_report.json'
    report = {x['session_id']: x for x in json.loads(report_path.read_text())} if report_path.exists() else {}
    for r in rows:
        dest = OUT/r['cohort'].split(' (')[0].replace(' ', '_')/f"{r['session_id']}.h5"
        if dest.exists() and not a.overwrite and not a.check: print('skip (exists)', r['session_id']); continue
        try:
            res = convert(r, a.check); print('OK  ', json.dumps(res), flush=True)
        except Exception as e:
            res = dict(session_id=r['session_id'], error=f'{type(e).__name__}: {e}'); print('FAIL', r['session_id'], res['error'], flush=True)
            traceback.print_exc(limit=1)
        report[r['session_id']] = res
        report_path.write_text(json.dumps(list(report.values()), indent=1))
