"""Can breathing decode relative rank?  Jan + Jul 2026 cagemate recordings.  Manuscript version (v2).

Question  In a cagemate recording, does the subject's breathing tell whether it OUTRANKS its cagemate or is OUTRANKED?
          Relative rank from src/r34_ranks.py (tube-test sheet), never from file names.
Data      Shared H5s (01_shared_h5/): breaths = respiration/breaths/reviewed (hand review, written by
          scripts/apply_review_to_shared_h5.py; review_stage FINAL or returned), bad signal = reviewer_excluded.
Windows   Fixed 3 s, non-overlapping, over all cagemate time; a window is used only if it has no bad signal.
          (PI decision: rank is a state, so fixed windows, not behavior bouts.)
Tags      Cleaned BORIS (03_boris/cleaned/): 'investigation' if either mouse sniffs (face/body/anogenital), allogrooms or
          chases during any part of the window. MAIN analysis = windows with no investigation.
Features  rate; inter-breath interval mean / SD / CV / min / max; fraction of intervals < 125 ms (> 8 Hz);
          amplitude CV; amplitude mean and SD relative to the recording's median amplitude (raw amplitude depends on gain).
Control   Locomotion: both mice's speed (body lengths/s, SLEAP, identity-free) per window: mean speed, fraction moving.
Cohort    Features standardized within each cohort with TRAINING recordings only; training weights give every
          cohort x label cell equal weight (PI: cohort-balanced). Within-cohort decoders reported too.
Testing   Outer: leave one cage out (cage = cohort + cage number). Inner (tuning, training cages only): 3-fold grouped by cage,
          scored by recording-level AUC. Logistic: C in {0.01, 0.1, 1, 10}. RBF SVM: C in {0.3, 3} x gamma in {0.5, 2}/n_features.
          A recording's score = mean decision value of its windows. Main number = recording-level AUC over all held-out recordings.
CI        95% cluster bootstrap over cages (2,000 resamples of the held-out recording scores).
Null      Recording labels shuffled within cohort; the WHOLE pipeline (including tuning) is rerun; p = (1 + #null >= obs)/(1 + n).

Usage (biopipeline-env):  python scripts/rank_decoder_2026.py [--perm 200] [--quick]
Outputs:  07_rank_decoder/ (windows table, results, held-out scores, null distributions, feature importance, run_info.json)
"""
from pathlib import Path
import sys, json, argparse, datetime, platform, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)       # 'mean of empty slice' on frames where SLEAP has no mouse
import numpy as np, pandas as pd, h5py
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import sklearn

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from r34_ranks import session_info, CAGEMATE_SESSIONS

LAB = Path('C:/Users/sjs93/UF Dropbox/Sequioa Smith/Padilla-Coreano Lab')
MS = LAB/'2026/respiration_valence_isolation_manuscript_SSmith'
H5, BOR, OUT = MS/'01_shared_h5', MS/'03_boris/cleaned', MS/'07_rank_decoder'
SLEAP = {'Jul 2026': [MS/'04_sleap/id_corrected/Cagemate_B_Jul2026'],
         'Jan 2026': [MS/'04_sleap/round2_retrack_hpg/round2b_masked/tracked', LAB/'2025/R34_valence_SLEAP_retrack/corrected']}
FS, WIN = 100, 300
COHORT_DIR = {'Jan 2026': 'Cagemate_C', 'Jul 2026': 'Cagemate_B'}
EXCLUDE = {'10_2_d_CM_10_1_s_20260729_114455': 'unusable respiration (EXCLUSIONS_LOG)',
           '11_3_s_CM_11_4_d_20260729_122040': 'unusable respiration (EXCLUSIONS_LOG)',
           '13_2_d_CM_13_1_s_20260729_181321': 'second 13.2 recording (first one used until decided)'}
INVESTIGATION = {'facial sniffing', 'body sniffing', 'anogenital sniffing', 'allogrooming', 'chasing'}
BREATH = ['rate', 'ibi_mean', 'ibi_sd', 'ibi_cv', 'ibi_min', 'ibi_max', 'frac_fast', 'amp_cv', 'amp_rel_mean', 'amp_rel_sd']
SPEED = ['speed_mean', 'frac_moving']
GRID = {'logistic': [dict(C=c) for c in (0.01, 0.1, 1, 10)],
        'svm_rbf': [dict(C=c, g=g) for c in (0.3, 3) for g in (0.5, 2)]}


# ----------------------------------------------------------------------------------------------- data
def pair_speed(sid, cohort, frame_t):
    """Per-frame speed of both mice (body lengths/s), identity-free; None if no SLEAP file."""
    stem = sid
    f = next((p for d in SLEAP[cohort] for p in sorted(d.glob(f'{stem}*.analysis.h5')) + sorted(d.glob(f'{stem}*.h5'))
              if ('noreflect' in p.name or 'round2b' in str(p) or cohort == 'Jul 2026')), None)
    if f is None: return None
    with h5py.File(f, 'r') as h: X = h['tracks'][:].transpose(3, 0, 2, 1)        # frames x 2 x 8 nodes x xy
    gap = len(frame_t) - len(X)                     # SLEAP often stops 1-2 frames before the video ends
    if not 0 <= gap <= 5: return None
    if gap: X = np.concatenate([X, np.full((gap,) + X.shape[1:], np.nan, X.dtype)])
    bl = np.nanmedian(np.linalg.norm(X[:, :, 0] - X[:, :, 7], axis=2))
    c = np.nanmean(X[:, :, 4:7], axis=2); fps = 1/np.median(np.diff(frame_t)); k = max(1, int(round(.2*fps)))
    a, b = c[:-k], c[k:]
    same, swap = np.linalg.norm(a - b, axis=2), np.linalg.norm(a - b[:, ::-1], axis=2)
    d = np.where((swap.sum(1) < same.sum(1))[:, None], swap, same) / (k/fps) / bl
    d[d > 8] = np.nan
    return np.r_[d, np.full((k, 2), np.nan)]


def windows_for(sid):
    info = session_info(sid)
    with h5py.File(H5/COHORT_DIR[info['cohort']]/f'{sid}.h5', 'r') as h:
        g = h.get('respiration/breaths/reviewed')
        if g is None or 'breath_index' not in g: return None, 'no reviewed breaths in H5'
        peaks = g['breath_index'][:]; bad = g['reviewer_excluded'][:].astype(bool); stage = g.attrs.get('review_stage', '')
        y = h['respiration/preprocessed/pressure'][:].astype(float); t = h['respiration/preprocessed/time_s'][:]
        b = h['phases/boundaries_s'][:]; lab = [x.decode() if isinstance(x, bytes) else x for x in h['phases/labels'][:]]
        ft = h['video/frame_time_s'][:] if 'video' in h else None
    n = len(y); ok = ~bad
    amp = np.array([y[p] - y[q:p+1].min() for q, p in zip(np.r_[max(0, peaks[0]-50), peaks[:-1]], peaks)])
    amp_med = np.median(amp)
    cm0 = int(np.searchsorted(t, b[lab.index('cagemate')])) if 'cagemate' in lab else 0
    bo = pd.read_csv(BOR/COHORT_DIR[info['cohort']]/f'{sid}_boris.csv')
    bo = bo[bo.behavior.isin(INVESTIGATION) & bo.start_recording_s.notna()]
    sp = pair_speed(sid, info['cohort'], ft) if ft is not None else None
    rows = []
    for s0 in range(cm0, n - WIN + 1, WIN):
        if not ok[s0:s0+WIN].all(): continue
        i = np.flatnonzero((peaks >= s0) & (peaks < s0 + WIN)); ibi = np.diff(peaks[i]) / FS; a = amp[i]
        t0, t1 = t[s0], t[s0 + WIN - 1]
        ov = bo[(bo.start_recording_s < t1) & (bo.stop_recording_s > t0)]
        tag = 'none' if not len(ov) else ('both' if set(ov.actor) == {'subject', 'agent'} else
                                         'subject investigating' if 'subject' in set(ov.actor) else 'being investigated')
        r = dict(session_id=sid, cohort=info['cohort'], cage=f"{info['cohort']} cage {info['cage']}", subject=info['subject'],
                 partner=info['partner'], relative=info['relative'], sex=info['sex'], review_stage=stage, start_s=t0 - t[cm0], tag=tag,
                 rate=len(i) / (WIN / FS),
                 ibi_mean=ibi.mean() if len(ibi) else np.nan, ibi_sd=ibi.std() if len(ibi) > 1 else np.nan,
                 ibi_cv=ibi.std() / ibi.mean() if len(ibi) > 1 else np.nan, ibi_min=ibi.min() if len(ibi) else np.nan,
                 ibi_max=ibi.max() if len(ibi) else np.nan, frac_fast=(ibi < .125).mean() if len(ibi) else np.nan,
                 amp_cv=a.std() / a.mean() if len(a) > 1 else np.nan, amp_rel_mean=a.mean() / amp_med if len(a) else np.nan,
                 amp_rel_sd=a.std() / amp_med if len(a) > 1 else np.nan, speed_mean=np.nan, frac_moving=np.nan)
        if sp is not None:
            f = (ft >= t0) & (ft < t1); v = sp[f].ravel(); v = v[np.isfinite(v)]
            if len(v) > 10: r.update(speed_mean=v.mean(), frac_moving=(v > .3).mean())
        rows.append(r)
    return rows, stage


# ----------------------------------------------------------------------------------------------- model
def prep(train, test, feats):
    """Per-cohort standardization from training rows only; missing values -> 0 (= the training mean)."""
    Xtr, Xte = train[feats].to_numpy(float).copy(), test[feats].to_numpy(float).copy()
    ctr, cte = train.cohort.to_numpy(), test.cohort.to_numpy()
    for c in np.unique(ctr):
        m, s = np.nanmean(Xtr[ctr == c], 0), np.nanstd(Xtr[ctr == c], 0) + 1e-9
        Xtr[ctr == c] = (Xtr[ctr == c] - m) / s; Xte[cte == c] = (Xte[cte == c] - m) / s
    return np.nan_to_num(Xtr), np.nan_to_num(Xte)


def weights(d):
    k = d.cohort + '|' + d.y.astype(str); return (1.0 / k.map(k.value_counts())).to_numpy() * len(d) / k.nunique()


def make(model, p, nfeat):
    return LogisticRegression(C=p['C'], max_iter=3000) if model == 'logistic' else SVC(kernel='rbf', C=p['C'], gamma=p['g'] / nfeat)


def rec_auc(d, score):
    r = pd.DataFrame(dict(s=d.session_id.to_numpy(), y=d.y.to_numpy(), v=score)).groupby('s').agg(y=('y', 'first'), v=('v', 'mean'))
    return roc_auc_score(r.y, r.v) if r.y.nunique() == 2 else np.nan


def fit_predict(tr, te, model, feats, tune=True):
    """Tune on training cages (3-fold grouped by cage, recording-level AUC), refit, return test decision values + chosen params."""
    best = GRID[model][0] if not tune else None
    if tune:
        cages = np.array(sorted(tr.cage.unique())); folds = np.array_split(np.random.default_rng(0).permutation(cages), 3)
        scores = []
        for p in GRID[model]:
            s = []
            for f in folds:
                itr, ite = tr[~tr.cage.isin(f)], tr[tr.cage.isin(f)]
                if itr.y.nunique() < 2 or ite.y.nunique() < 2: continue
                Xa, Xb = prep(itr, ite, feats); m = make(model, p, len(feats)).fit(Xa, itr.y, sample_weight=weights(itr))
                s.append(rec_auc(ite, m.decision_function(Xb)))
            scores.append(np.nanmean(s) if s else -1)
        best = GRID[model][int(np.argmax(scores))]
    Xa, Xb = prep(tr, te, feats); m = make(model, best, len(feats)).fit(Xa, tr.y, sample_weight=weights(tr))
    return m.decision_function(Xb), best, m, Xb


def run(D, model, feats, labels=None, importance=False):
    """Leave one cage out. Returns per-window held-out scores (and permutation importance if asked)."""
    D = D.copy()
    if labels is not None: D['y'] = D.session_id.map(labels)
    score = np.full(len(D), np.nan); chosen, imp = [], []
    for cage in sorted(D.cage.unique()):
        tr, te = D[D.cage != cage], D[D.cage == cage]
        if tr.y.nunique() < 2:                          # can happen with shuffled labels in small subsets (Jul 2026 only):
            score[(D.cage == cage).to_numpy()] = 0.; continue   # no model can be fit -> neutral score (= chance), never NaN
        v, p, m, Xb = fit_predict(tr, te, model, feats); score[(D.cage == cage).to_numpy()] = v; chosen.append(p)
        if importance:                                  # drop in recording-level AUC on THIS held-out cage is unstable;
            rng = np.random.default_rng(1)               # so store held-out decision values with each feature shuffled
            for j, f in enumerate(feats):
                Xp = Xb.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); imp.append(dict(cage=cage, feature=f, idx=np.flatnonzero((D.cage == cage).to_numpy()), v=m.decision_function(Xp)))
    D['score'] = score
    if importance:
        out = []
        base = rec_auc(D, score)
        for f in feats:
            s = score.copy()
            for r in [x for x in imp if x['feature'] == f]: s[r['idx']] = r['v']
            out.append(dict(feature=f, auc_drop=base - rec_auc(D, s)))
        return D, chosen, pd.DataFrame(out)
    return D, chosen


def recording_table(D):
    return D.groupby('session_id').agg(y=('y', 'first'), score=('score', 'mean'), cohort=('cohort', 'first'), cage=('cage', 'first'),
                                       subject=('subject', 'first'), partner=('partner', 'first'), relative=('relative', 'first'),
                                       n_windows=('score', 'size'))


def cage_bootstrap(R, n=2000, seed=0):
    rng = np.random.default_rng(seed); cages = R.cage.unique(); out = []
    for _ in range(n):
        s = pd.concat([R[R.cage == c] for c in rng.choice(cages, len(cages))])
        if s.y.nunique() == 2: out.append(roc_auc_score(s.y, s.score))
    return np.percentile(out, [2.5, 97.5])


def shuffled(R, seed):
    rng = np.random.default_rng(seed); lab = {}
    for c, g in R.groupby('cohort'): lab.update(dict(zip(g.index, rng.permutation(g.y.to_numpy()))))
    return lab


def analysis(name, D, model, feats, nperm):
    D = D.reset_index(drop=True)
    Dw, chosen, imp = run(D, model, feats, importance=True)
    R = recording_table(Dw); obs = roc_auc_score(R.y, R.score)
    null = Parallel(n_jobs=-1)(delayed(lambda s: rec_auc(*(lambda d: (d, d.score.to_numpy()))(run(D, model, feats, shuffled(R, s))[0])))(s)
                               for s in range(nperm))
    null = np.array(null); lo, hi = cage_bootstrap(R)
    res = dict(analysis=name, model=model, features='+'.join(feats) if len(feats) < 4 else f'{len(feats)} features',
               rec_auc=obs, ci_low=lo, ci_high=hi, rec_acc=((R.score > 0) == (R.y == 1)).mean(),
               win_auc=roc_auc_score(Dw.y, Dw.score), p=(1 + (null >= obs).sum()) / (1 + nperm), null_mean=null.mean(),
               null_95=np.percentile(null, 95), n_perm=nperm, n_rec=len(R), n_mice=R.subject.nunique(), n_cages=R.cage.nunique(), n_win=len(Dw),
               params=json.dumps(pd.Series([json.dumps(c) for c in chosen]).value_counts().to_dict()))
    for c in ['Jan 2026', 'Jul 2026']:
        r = R[R.cohort == c]; res[f'auc_{c[:3]}'] = roc_auc_score(r.y, r.score) if r.y.nunique() == 2 else np.nan
    return res, R.assign(analysis=name, model=model), null, imp.assign(analysis=name, model=model)


# ----------------------------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--perm', type=int, default=200); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); OUT.mkdir(exist_ok=True); nperm = 10 if a.quick else a.perm
    sids = [s for s, v in CAGEMATE_SESSIONS.items() if v[0] in COHORT_DIR and s not in EXCLUDE]
    rows, stages = [], {}
    for s in sids:
        w, st = windows_for(s); stages[s] = st
        if w is None: print('skip', s, st); continue
        rows += w
    W = pd.DataFrame(rows); W = W[W.relative.isin(['higher', 'lower'])].copy()
    W['y'] = (W.relative == 'lower').astype(int)                       # 1 = subject is outranked
    W.to_csv(OUT/'windows_2026.csv', index=False)
    print(W.groupby(['cohort', 'relative']).session_id.nunique().rename('recordings'), W.tag.value_counts(), sep='\n')
    none = W[W.tag == 'none']; sp = none[none.speed_mean.notna()]
    plan = [('Main: no-investigation windows', none, 'logistic', BREATH, nperm),
            ('Main: no-investigation windows', none, 'svm_rbf', BREATH, nperm),
            ('All windows', W, 'logistic', BREATH, nperm),
            ('All windows', W, 'svm_rbf', BREATH, nperm),
            ('Jan 2026 only (no-investigation)', none[none.cohort == 'Jan 2026'], 'svm_rbf', BREATH, nperm),
            ('Jul 2026 only (no-investigation)', none[none.cohort == 'Jul 2026'], 'svm_rbf', BREATH, nperm),
            ('Rate only (no-investigation)', none, 'logistic', ['rate'], nperm),
            ('Control: locomotion only', sp, 'svm_rbf', SPEED, nperm),
            ('Breathing, same windows as locomotion', sp, 'svm_rbf', BREATH, nperm),
            ('Breathing + locomotion', sp, 'svm_rbf', BREATH + SPEED, nperm)]
    results, recs, nulls, imps = [], [], {}, []
    for name, D, model, feats, n in plan:
        res, R, null, imp = analysis(name, D, model, feats, n); results.append(res); recs.append(R); imps.append(imp)
        nulls[f'{name} | {model}'] = null.tolist()
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != 'params'}, flush=True)
        pd.DataFrame(results).to_csv(OUT/'decoder_results.csv', index=False)
    pd.concat(recs).to_csv(OUT/'heldout_recording_scores.csv')
    pd.concat(imps).to_csv(OUT/'feature_importance.csv', index=False)
    (OUT/'null_distributions.json').write_text(json.dumps(nulls))
    (OUT/'run_info.json').write_text(json.dumps(dict(
        run_at=datetime.datetime.now().isoformat(timespec='seconds'), script='scripts/rank_decoder_2026.py', n_perm=nperm,
        python=platform.python_version(), sklearn=sklearn.__version__, excluded=EXCLUDE, grid={k: v for k, v in GRID.items()},
        review_stage=stages, n_windows=int(len(W)), tags=W.tag.value_counts().to_dict()), indent=1))
