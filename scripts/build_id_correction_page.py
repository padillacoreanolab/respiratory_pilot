"""Write SLEAP_tracked/START_HERE.html in the lab Dropbox: assignments + step-by-step ID correction.

Status per video is read from the files each time this runs (to_correct/ = ready, corrected/ = done).
Rerun after tracking, and whenever you want the table updated:  python scripts/build_id_correction_page.py
"""
from pathlib import Path
import html, csv
import h5py

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT/'outputs/aim1_summer2026_ranked_recs'
OUT = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2026/BLA_ChR_resp_cohort2/BLA_ChR_resp_CM_hc/SLEAP_tracked')
# Reviewer per cage.
ASSIGN = {'10': 'Samhitha', '14': 'Samhitha', '11': 'Baano', '13': 'Marissol'}

summary = {}
if (OUT/'tracking_summary.csv').exists():
    summary = {r['video']: r for r in csv.DictReader(open(OUT/'tracking_summary.csv'))}
rows = []
for h5 in sorted(H5.glob('*.h5')):
    stem = h5.stem
    with h5py.File(h5, 'r') as h:
        a = h.attrs; cage = str(a['cage_id'])
        subj = f"Subject {a['subject_id']} (rank {int(a['subject_rank'])})"; mate = f"Cagemate {a['partner_id']} (rank {int(a['partner_rank'])})"
    todo = OUT/'to_correct'/f'{stem}.1.tracked.slp'; done = OUT/'corrected'/f'{stem}.1.idcorrected.slp'
    h5out = OUT/'corrected'/f'{stem}.1.analysis.h5'
    status = ('Done (corrected + exported)' if done.exists() and h5out.exists() else 'Corrected, export missing' if done.exists()
              else 'Ready to correct' if todo.exists() else 'Waiting for tracking')
    s = summary.get(stem, {})
    rows.append((ASSIGN[cage], stem, subj, mate, s.get('close_contact_s', ''), s.get('suggestions', ''), status))
rows.sort(key=lambda r: (r[0], r[1]))
tr = '\n'.join(f'<tr><td><b>{html.escape(r[0])}</b></td><td><code>{html.escape(r[1])}</code></td><td>{html.escape(r[2])}</td>'
               f'<td>{html.escape(r[3])}</td><td>{r[4]}</td><td>{r[5]}</td><td>{html.escape(r[6])}</td></tr>' for r in rows)

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLEAP ID Correction</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#5f5f66;--line:#d9d9de;--panel:#f5f5f7;--accent:#1677b8;--warn:#a15c00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#17171a;--fg:#ececf0;--muted:#a0a0a8;--line:#34343a;--panel:#212126;--accent:#5aa9e6;--warn:#e0a24a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 8px;border-top:1px solid var(--line);padding-top:16px}}
p,li{{color:var(--fg)}} .muted{{color:var(--muted)}} code{{background:var(--panel);padding:1px 5px;border-radius:4px;font-size:13px}}
kbd{{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 6px;font-size:13px}}
.tbl{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:var(--panel)}}
.note{{border-left:3px solid var(--warn);background:var(--panel);padding:8px 12px;border-radius:4px}}
ol li,ul li{{margin:4px 0}}
</style></head><body><main>
<h1>SLEAP ID correction: cohort 2 ranked cagemates (July 29, 2026)</h1>
<p class="muted">Each video was tracked by our SLEAP model. The skeletons (body points) are already good. Your job is only to make sure each mouse keeps the <b>right name</b> for the whole social phase: the computer sometimes swaps the two mice when they touch. You are not re-placing body points.</p>

<h2>Your videos</h2>
<div class="tbl"><table>
<tr><th>Reviewer</th><th>Video</th><th>Subject (has the tether)</th><th>Cagemate</th><th>Close-contact s</th><th>Checkpoints</th><th>Status</th></tr>
{tr}
</table></div>
<p class="muted">"Checkpoints" = frames in the Labeling Suggestions list (likely swap spots). Status is read from the files in this folder.</p>

<h2 id="naming">Naming your files</h2>
<p>Keep the video name exactly as it is and change <b>only the ending</b>. Everything you make goes in <code>SLEAP_tracked/corrected/</code>. Example for video 11.4:</p>
<div class="tbl"><table>
<tr><th>File</th><th>Name (type this in the File name box)</th></tr>
<tr><td>You open (in <code>to_correct/</code>)</td><td><code>11_4_d_CM_11_3_s_20260729_125055.1.tracked.slp</code></td></tr>
<tr><td>Your corrected copy (Save As)</td><td><code>11_4_d_CM_11_3_s_20260729_125055.1.idcorrected.slp</code></td></tr>
<tr><td>Exported analysis file</td><td><code>11_4_d_CM_11_3_s_20260729_125055.1.analysis.h5</code></td></tr></table></div>

<h2>Before you start (once)</h2>
<ol>
<li>Install SLEAP <b>1.3.3</b> if your computer doesn't have it (Miniforge prompt):<br><code>conda create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3</code><br>Then open it with <code>conda activate sleap</code> and <code>sleap-label</code>.</li>
<li>In File Explorer, right-click this Dropbox folder <code>BLA_ChR_resp_CM_hc</code> and choose <b>Make available offline</b>. The videos are ~200 MB each and SLEAP needs them on your computer.</li>
</ol>

<h2>Open your video</h2>
<ol>
<li>In SLEAP: <b>File → Open Project</b> and choose your file in <code>SLEAP_tracked/to_correct/</code> (e.g. <code>11_3_s_CM_11_4_d_20260729_122040.1.tracked.slp</code>).</li>
<li>If SLEAP says it <b>can't find the video</b>, click to locate it and pick <code>BLA_ChR_resp_CM_hc/&lt;same name&gt;.rec/&lt;same name&gt;.1.mp4</code> (same name as your file, without <code>.tracked.slp</code>).</li>
<li><b>Right away: File → Save As</b>. In the Save window, first go to the folder <code>SLEAP_tracked/corrected/</code>, then in the <b>File name</b> box type only the name, ending in <code>.idcorrected.slp</code> (see <a href="#naming">Naming your files</a>). Work only in that copy; never edit files in <code>to_correct/</code>.</li>
<li>Set up the view:
  <ul>
  <li><b>View → Color Predicted Instances</b> (on): each mouse gets its own color.</li>
  <li><b>Tracks → Seekbar Header → Min Centroid Proximity</b>: the timeline then shows when the mice are close (where swaps happen).</li>
  <li>Open the <b>Instances</b> tab (bottom right). Click a mouse to see its name in the <b>Track</b> column; a box on the video also shows <i>Track: …</i>.</li>
  </ul></li>
</ol>

<h2>Who is who</h2>
<ul>
<li><b>Subject</b> = the mouse with the <b>tether</b> (the cable from its head). It is in the cage alone at the start.</li>
<li><b>Cagemate</b> = the mouse without the tether, put into the cage partway through.</li>
<li>Each track name also gives the mouse ID and its rank, e.g. <i>Subject 11.3 (rank 3)</i>, <i>Cagemate 11.4 (rank 1)</i>. Don't rename tracks.</li>
<li>Only the stretch from 30 s before the cagemate is added to the end was tracked. Frames before that have no skeletons; that's expected.</li>
</ul>

<h2>Correct the IDs</h2>
<ol>
<li><b>The hand.</b> The first checkpoint is when the cagemate is put in. The model sometimes puts a skeleton on the experimenter's <b>hand</b>. Click any skeleton on a hand and press <kbd>Delete</kbd>, frame by frame, until the cagemate is fully in the cage.</li>
<li><b>Checkpoints.</b> Open <b>Labeling Suggestions</b> and press <kbd>Space</kbd> to jump to each checkpoint (a close contact, or a moment a mouse's skeleton jumped). At each one, step forward with <kbd>→</kbd> (1 frame) or <kbd>Ctrl</kbd>+<kbd>→</kbd> (a few frames) through the contact and check the <b>Subject</b> color stays on the tethered mouse.</li>
<li><b>Fixing a swap.</b> Go to the <b>first frame where the names are wrong</b> and press <kbd>Ctrl</kbd>+<kbd>T</kbd> (Tracks → Transpose Instance Tracks). This swaps the two names <b>from this frame to the end of the video</b>.
  <ul>
  <li>If the names flip back wrong again later, press <kbd>Ctrl</kbd>+<kbd>T</kbd> again at the first wrong frame there.</li>
  <li>A swap that lasts only a few frames needs <b>two</b> presses: one at its first wrong frame, one at the first frame after it that is wrong again.</li>
  <li>Always go forward in time. Fixing an earlier frame after a later one undoes the later fix.</li>
  </ul></li>
<li><b>Whole pass.</b> After the checkpoints, go through the whole social phase once with <kbd>Ctrl</kbd>+<kbd>→</kbd>, watching the colors. Swaps can happen without a checkpoint.</li>
<li><b>Other problems.</b> Two skeletons on one mouse, or a skeleton on nothing: click the wrong one and press <kbd>Delete</kbd>. A mouse missing for a few frames: leave it. A body point slightly off: leave it.</li>
<li>Save often with <kbd>Ctrl</kbd>+<kbd>S</kbd>.</li>
</ol>

<h2>When you finish a video</h2>
<ol>
<li><kbd>Ctrl</kbd>+<kbd>S</kbd> one last time.</li>
<li><b>File → Export Analysis HDF5… → Current video</b>, saved in <code>corrected/</code> with the same name ending in <code>.1.analysis.h5</code> (e.g. <code>11_4_d_CM_11_3_s_20260729_125055.1.analysis.h5</code>).</li>
<li>In the <b>Notion</b> SLEAP ID Correction table, open your video's row, set <b>Tracking Status</b> to <i>Uploaded to Dropbox</i>, and write in the <b>Notes</b> column: your name, the date, how long it took, and anything odd (frame numbers you weren't sure about).</li>
<li>Tell Sequioa it's done.</li>
</ol>
<p class="note">If you're unsure about anything, ask Sequioa.</p>
</main></body></html>
'''
OUT.mkdir(parents=True, exist_ok=True); (OUT/'to_correct').mkdir(exist_ok=True); (OUT/'corrected').mkdir(exist_ok=True)
(OUT/'START_HERE.html').write_text(page, encoding='utf-8')
print(OUT/'START_HERE.html')
for r in rows: print(f'{r[0]:9s} {r[1]:34s} {r[6]}')
