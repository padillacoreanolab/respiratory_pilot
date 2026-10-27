"""Build a BLIND labeling set of candidate breath peaks from held-out data.

  python make_labeling_set.py <out_dir> [n_per_class=60]

Samples (seeded) from rank evaluation minutes + valence recordings:
  legacy_only   : peaks only the legacy detector reports
  adaptive_only : peaks only the adaptive detector reports
  both          : peaks both report (controls)
  adaptive_reject: candidates the adaptive detector rejected (low_prominence etc.)
Each candidate is drawn without any detector markings (only a neutral marker at the
candidate time), shuffled, and given an anonymous id. The source class is stored in
labeling/answer_key.csv, which the labeler should not open.
Open labeling/label_breaths.html in a browser, label, click "Download CSV", save it as
labeling/labels.csv, then run score_labels.py.
"""
import base64
import glob
import io
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from real_common import FS, match_idx  # noqa: E402

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LAB = os.path.join(OUT, "labeling")
os.makedirs(LAB, exist_ok=True)
TOL = int(0.025 * FS)
rng = np.random.default_rng(7)
EXCLUDE = {"RI1_5_13", "RI2_5_13", "BLRI_5_13"}  # documented drops (docs/EXCLUSIONS_LOG.md)

cands = []
for p in sorted(glob.glob(os.path.join(OUT, "recordings", "*.npz"))):
    key = os.path.basename(p)[:-4]
    if key in EXCLUDE:
        continue
    d = np.load(p)
    ok = d["evalmask"] & d["valid"]
    pa, pl = d["peaks_a"], d["peaks_l"]
    pa, pl = pa[ok[pa]], pl[ok[pl]] if len(pl) else pl
    la, al = match_idx(pl, pa, TOL), match_idx(pa, pl, TOL)
    for idx, cls in [(pl[~la], "legacy_only"), (pa[~al], "adaptive_only"), (pa[al], "both")]:
        for i in idx:
            cands.append((key, int(i), cls))
df =pd.DataFrame(cands, columns=["key", "idx", "cls"])
# keep candidates away from recording edges
df = df[df.idx > FS].copy()
picked = []
for cls in ["legacy_only", "adaptive_only", "both"]:
    sub = df[df.cls == cls]
    picked.append(sub.sample(min(N, len(sub)), random_state=int(rng.integers(1e9))))
# adaptive-rejected near misses (fresh fit on the cached signals)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_breath_detection import detect_breaths  # noqa: E402
from real_common import all_recordings, load_signal  # noqa: E402
recs = all_recordings()
rows = []
for key in rng.permutation(sorted(df.key.unique()))[:12]:
    sig, t = load_signal(key, recs[key])
    r = detect_breaths(sig, FS, time_s=t)
    rj = r.rejected[r.rejected.reason.isin(["low_prominence", "shallow_trough", "too_close"])]
    ev = np.load(os.path.join(OUT, "recordings", f"{key}.npz"))["evalmask"]
    rj = rj[ev[rj.idx.astype(int)] & (rj.prominence >= 0.5 * rj.required)]
    for i in rj.idx.sample(min(6, len(rj)), random_state=1):
        rows.append((key, int(i), "adaptive_reject"))
rows = pd.DataFrame(rows, columns=["key", "idx", "cls"])
picked.append(rows.sample(min(N, len(rows)), random_state=2))
lab = pd.concat(picked, ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
lab["item_id"] = [f"item_{i:03d}" for i in range(len(lab))]
lab[["item_id", "key", "idx", "cls"]].to_csv(os.path.join(LAB, "answer_key.csv"), index=False)

html_items = []
cache = {}
for r in lab.itertuples():
    if r.key not in cache:
        cache[r.key] = np.load(os.path.join(OUT, "recordings", f"{r.key}.npz"))["sig"]
    sig = cache[r.key]
    s0, s1 = max(0, r.idx - int(1.0 * FS)), min(len(sig), r.idx + int(1.0 * FS))
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(np.arange(s0, s1) / FS - r.idx / FS, sig[s0:s1], color="#167fa0", lw=1)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.plot(0, sig[r.idx], "D", mfc="none", mec="k", ms=10)
    ax.set_xlabel("seconds relative to candidate")
    ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=70)
    plt.close(fig)
    html_items.append((r.item_id, base64.b64encode(buf.getvalue()).decode()))

blocks = "\n".join(
    f'<div class="it" id="{i}"><div class="h">{i}</div><img src="data:image/png;base64,{b}">'
    f'<div class="b">' + "".join(
        f'<label><input type="radio" name="{i}" value="{v}"> {v}</label>' for v in ["breath", "not_breath", "unsure"]) +
    "</div></div>" for i, b in html_items)
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Breath labeling</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px;background:#fff;color:#111}}
.it{{border-bottom:1px solid #ddd;padding:8px 0}} .h{{font-weight:bold}} img{{max-width:100%}}
label{{margin-right:18px}} #bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #333}}</style></head>
<body><div id="bar"><b>Is the diamond-marked point the peak of a real breath (inspiratory peak)?</b>
<button onclick="dl()">Download CSV</button> <span id="cnt"></span></div>{blocks}
<script>
const K='breath_labels_v1';let S={{}};try{{S=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function cnt(){{document.getElementById('cnt').textContent=Object.keys(S).length+' / {len(html_items)} labeled'}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(S[r.name]===r.value)r.checked=true;
 r.addEventListener('change',()=>{{S[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(S))}}catch(e){{}};cnt()}})}});cnt();
function dl(){{let t='item_id,label\\n'+Object.entries(S).map(([k,v])=>k+','+v).join('\\n');
 let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{{type:'text/csv'}}));a.download='labels.csv';a.click()}}
</script></body></html>"""
open(os.path.join(LAB, "label_breaths.html"), "w", encoding="utf-8").write(html)
print(lab.cls.value_counts().to_string())
print("wrote", os.path.join(LAB, "label_breaths.html"))
