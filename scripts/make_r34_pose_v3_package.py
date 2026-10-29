"""Build a SLEAP training package (v3) for the R34 valence / isolation / cagemate A videos from Tanish's labels.

Run with the sleap env (Python 3.7). Only reads Tanish's projects; writes a NEW folder.
Labels: labels.c57final.slp (RI1, 701 frames; trained his n=701 models) + ECG_cohort1.v006.slp (RI2, 787 frames;
n=787 models). Video paths are moved from his Downloads copies to the original .1.mp4 files (labels checked by eye
on 30 frames, all on the mice). Left out: the Clipchamp-edited RI1 3.5 clip and mouse 5.13 (not in the paper).
A fixed 10% of frames per video is held out (val.pkg.slp) so old and new models are scored on the same frames.

Changes from Tanish's configs (his: rotation -180..-180 = no rotation augmentation; every frame resized to
640x480 because one 640x480 clip was in the training set):
  rotation -180..+180, no resize (all videos 1280x720), centroid input scaling 0.5, body-point crop 256 px
  (mice up to ~220 px nose to tail base), body-point max stride 32 (receptive field ~150 px), filters 24,
  sigma 2.5, early stopping 15, anchor 'head' (as his models).
Also writes hipergator/ (sbatch: train both models, then score old vs new on the held-out frames).

Usage (sleap env):  python scripts/make_r34_pose_v3_package.py
"""
from pathlib import Path
import json, random
import sleap

TANISH = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2026/tanish')
ROOT = Path(__file__).resolve().parents[1]
OUT = Path('C:/Users/sjs93/SLEAP_r34/training_package_v3')
PROFILES = Path(sleap.__file__).parent/'training_profiles'
HPG_DIR = '/blue/npadillacoreano/sequioasmith/r34_id_model'
ENV = '/blue/npadillacoreano/sequioasmith/conda_envs/sleap133'
OLD = {'RI1': ('250707_125107.centroid.n=701', '250707_130232.centered_instance.n=701'),
       'RI2': ('250626_150000.centroid.n=787.c57v.cd1', '250626_151314.centered_instance.n=787.C57v.CD1')}
ANCHOR = 'head'

man = json.loads((ROOT/'outputs/r34_shared/manifest.json').read_text())


def original(name):
    if '_2025' not in name: return None
    s = '2025' + name.split('_2025')[-1][:11]
    r = next((x for x in man if x['session_id'].endswith(s)), None)
    return next(Path(r['rec_dir']).glob('*.1.mp4'), None) if r else None


def config(profile, run_name, head, max_stride=None, scale=1.0):
    c = json.loads((PROFILES/profile).read_text())
    c['data']['labels']['training_labels'] = 'train.pkg.slp'
    c['data']['labels']['validation_labels'] = 'val.pkg.slp'
    c['data']['preprocessing']['input_scaling'] = scale
    c['data']['instance_cropping']['center_on_part'] = ANCHOR
    c['data']['instance_cropping']['crop_size'] = 256 if head == 'centered_instance' else None
    h = c['model']['heads'][head]; h['anchor_part'] = ANCHOR; h['sigma'] = 2.5
    u = c['model']['backbone']['unet']; u['filters'] = 24
    if max_stride: u['max_stride'] = max_stride
    a = c['optimization']['augmentation_config']
    a['rotate'] = True; a['rotation_min_angle'] = -180.; a['rotation_max_angle'] = 180.
    c['optimization']['early_stopping']['plateau_patience'] = 15
    c['outputs']['run_name'] = run_name; c['outputs']['runs_folder'] = 'models'; c['outputs']['save_visualizations'] = False
    return c


merged = None
for f in ['labels.c57final.slp', 'ECG_cohort1.v006.slp']:
    L = sleap.load_file(str(TANISH/f))
    for v in list(L.videos):
        o = original(Path(v.backend.filename).name)
        if o is None:
            for lf in list(L.find(v)): L.remove(lf)
            L.remove_video(v); print('left out', Path(v.backend.filename).name); continue
        v.backend.filename = str(o).replace('\\', '/')
    for lf in list(L.labeled_frames):                      # keep only hand-labeled mice
        lf.instances = list(lf.user_instances)
        if not lf.instances: L.remove(lf)
    if merged is None: merged = L
    else: merged.extend_from(L, unify=True)
random.seed(0)
train, val = [], []
for v in merged.videos:
    lfs = list(merged.find(v)); random.shuffle(lfs); k = max(1, round(.1*len(lfs)))
    val += lfs[:k]; train += lfs[k:]
OUT.mkdir(parents=True, exist_ok=True)
for name, lfs in (('train', train), ('val', val)):
    sleap.Labels(lfs).save(str(OUT/f'{name}.pkg.slp'), with_images=True, embed_all_labeled=True, embed_suggested=False)
print(f'train {len(train)} frames, val {len(val)} frames, {len(merged.videos)} videos')

cfgs = {'centroid.json': config('baseline.centroid.json', 'r34_v3.centroid', 'centroid', scale=0.5),
        'centered_instance.json': config('baseline_medium_rf.topdown.json', 'r34_v3.centered_instance', 'centered_instance', 32)}
for name, c in cfgs.items(): (OUT/name).write_text(json.dumps(c, indent=2))

hpg = OUT/'hipergator'; hpg.mkdir(exist_ok=True)
(hpg/'compare_models.py').write_text('''"""Score Tanish's old models and the new v3 models on the same held-out frames (val.pkg.slp)."""
import glob, json, sleap
from sleap.nn.evals import evaluate
val = sleap.load_file("val.pkg.slp")
runs = {"old_RI1": %s, "old_RI2": %s, "new_v3": sorted(glob.glob("models/r34_v3.*"))}
out = {}
for name, paths in runs.items():
    pred = sleap.load_model(paths, batch_size=4).predict(val)
    m = evaluate(val, pred)
    out[name] = {k: float(m[k]) for k in ("dist.p50", "dist.p90", "dist.p95", "oks_voc.mAP", "pck.mPCK")}
    print(name, out[name], flush=True)
json.dump(out, open("model_comparison.json", "w"), indent=1)
''' % (json.dumps([f'old_models/{m}' for m in OLD['RI1']]), json.dumps([f'old_models/{m}' for m in OLD['RI2']])))
(hpg/'train_hpg.sbatch').write_bytes((
    '#!/bin/bash\n'
    f'# Train the R34 v3 SLEAP models on one L4 GPU, then compare with the old models. Submit from {HPG_DIR}:\n'
    '#   sbatch hipergator/train_hpg.sbatch\n'
    '#SBATCH --job-name=sleap_r34_v3\n'
    '#SBATCH --account=npadillacoreano\n#SBATCH --qos=npadillacoreano\n'
    '#SBATCH --partition=hpg-turin\n#SBATCH --gpus=l4:1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=48gb\n'
    '#SBATCH --time=12:00:00\n#SBATCH --output=hipergator/train_%j.log\n\n'
    'set -euo pipefail\nmodule load conda\n'
    f'conda activate {ENV}\nnvidia-smi\ncd "$SLURM_SUBMIT_DIR"\n'
    'sleap-train centroid.json train.pkg.slp\n'
    'sleap-train centered_instance.json train.pkg.slp\n'
    'python hipergator/compare_models.py\n'
    'echo "FINISHED $(date)"\n').encode())
(OUT/'README.txt').write_text(
    'R34 SLEAP pose models v3 (valence RI1/RI2 labels from Tanish, fixed training settings)\n'
    'Upload this whole folder to ' + HPG_DIR + ' (include old_models/), then in HiPerGator Shell Access:\n'
    '  cd ' + HPG_DIR + '\n  sbatch hipergator/train_hpg.sbatch\n  squeue -u $USER      (check it is running)\n'
    'Results: models/r34_v3.*  and  model_comparison.json (old vs new on the same held-out frames).\n'
    'Built by scripts/make_r34_pose_v3_package.py.\n')
old = OUT/'old_models'; old.mkdir(exist_ok=True)
import shutil
for pair in OLD.values():
    for m in pair:
        d = old/m; d.mkdir(exist_ok=True)
        for fn in ('best_model.h5', 'training_config.json'): shutil.copy2(TANISH/'models'/m/fn, d/fn)
print('->', OUT)
