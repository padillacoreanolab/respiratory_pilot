"""Write R34_valence_SLEAP_retrack/START_HERE.html in the lab Dropbox: ID correction for the retracked valence videos.

These three RI videos were first tracked on re-encoded copies whose frames did not match the Trodes timestamps,
so they were retracked on the original videos (scripts/retrack_valence_sleap.ps1, Tanish's models and settings).
Status per video is read from the files each time this runs (to_correct/ = ready, corrected/ = done).
Usage (biopipeline-env): python scripts/build_valence_retrack_page.py
"""
from pathlib import Path
import html

OUT = Path('C:/Users/sjs93/UFL Dropbox/Sequioa Smith/Padilla-Coreano Lab/2025/R34_valence_SLEAP_retrack')
# (video stem, reviewer, experiment, other mouse)
VIDEOS = [('RI1_s4_7_p5_2_nRB3_20250621_150707', 'Samhitha', 'RI1 (juvenile intruder)', 'Juvenile: smaller black mouse, no headstage'),
          ('RI1_s4_8_p5_1_nRB3_20250621_165214', 'Samhitha', 'RI1 (juvenile intruder)', 'Juvenile: smaller black mouse, no headstage'),
          ('RI2_s2_3_p5_3_nRB3_20250622_110216', 'Samhitha', 'RI2 (CD1 agent)', 'Agent: white CD1 mouse')]

rows = []
for stem, who, exp, other in VIDEOS:
    todo = OUT/'to_correct'/f'{stem}.1.tracked.slp'; done = OUT/'corrected'/f'{stem}.1.idcorrected.slp'
    h5out = OUT/'corrected'/f'{stem}.1.analysis.h5'
    status = ('Done (corrected + exported)' if done.exists() and h5out.exists() else 'Corrected, export missing' if done.exists()
              else 'Ready to correct' if todo.exists() else 'Waiting for tracking')
    rows.append((who, stem, exp, other, status))
tr = '\n'.join(f'<tr><td><b>{html.escape(r[0])}</b></td><td><code>{html.escape(r[1])}</code></td><td>{html.escape(r[2])}</td>'
               f'<td>{html.escape(r[3])}</td><td>{html.escape(r[4])}</td></tr>' for r in rows)
ex = VIDEOS[0][0]

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Valence SLEAP Retrack</title>
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
<h1>SLEAP ID correction: valence resident-intruder videos (June 2025)</h1>
<p class="muted">These three videos were tracked again on the original recordings, with the same SLEAP model used for the other valence videos. The skeletons (body points) are already placed. Your job is only to make sure each mouse keeps the <b>right name</b> for the whole video: the computer sometimes swaps the two mice when they touch. You are not re-placing body points.</p>

<h2>Your videos</h2>
<div class="tbl"><table>
<tr><th>Reviewer</th><th>Video</th><th>Experiment</th><th>The other mouse</th><th>Status</th></tr>
{tr}
</table></div>
<p class="muted">Status is read from the files in this folder.</p>

<h2 id="naming">Naming your files</h2>
<p>Keep the video name exactly as it is and change <b>only the ending</b>. Everything you make goes in <code>R34_valence_SLEAP_retrack/corrected/</code>. Example for RI1 4.7:</p>
<div class="tbl"><table>
<tr><th>File</th><th>Name (type this in the File name box)</th></tr>
<tr><td>You open (in <code>to_correct/</code>)</td><td><code>{ex}.1.tracked.slp</code></td></tr>
<tr><td>Your corrected copy (Save As)</td><td><code>{ex}.1.idcorrected.slp</code></td></tr>
<tr><td>Exported analysis file</td><td><code>{ex}.1.analysis.h5</code></td></tr></table></div>

<h2>Before you start (once)</h2>
<ol>
<li>Use the same SLEAP <b>1.3.3</b> you used for the cohort 2 videos. Open it with <code>conda activate sleap</code> and then <code>sleap-label</code>.</li>
<li>In File Explorer (or Finder), right-click the Dropbox folder <code>R34_valence_SLEAP_retrack</code> and choose <b>Make available offline</b>. Wait for the green checks. The videos are ~200 MB each and SLEAP needs them on your computer.</li>
</ol>

<h2>Open your video</h2>
<ol>
<li>In SLEAP: <b>File → Open Project</b> and choose your file in <code>to_correct/</code>. Don't double-click the <code>.slp</code> file in File Explorer or Finder; the computer doesn't know to open it with SLEAP.</li>
<li>SLEAP will probably say it <b>can't find the video</b>. Don't cancel. Click the missing file, browse to <code>R34_valence_SLEAP_retrack/videos/</code>, pick the video with the same name (ending in <code>.1.mp4</code>), and click OK.</li>
<li>Check it loaded: you should see colored skeletons on the mice and <b>Labeled Frames</b> of about 18,000 at the bottom. If it says 0, close it and start again from step 1.</li>
<li><b>Right away: File → Save As</b>. In the Save window, first go to the folder <code>corrected/</code>, then in the <b>File name</b> box type only the name, ending in <code>.idcorrected.slp</code> (see <a href="#naming">Naming your files</a>). Work only in that copy; never edit files in <code>to_correct/</code>.</li>
<li>Set up the view:
  <ul>
  <li><b>View → Color Predicted Instances</b> (on): each mouse gets its own color.</li>
  <li><b>Tracks → Seekbar Header → Min Centroid Proximity</b>: the timeline then shows when the mice are close (where swaps happen).</li>
  <li>Open the <b>Instances</b> tab (bottom right). Click a mouse to see its name in the <b>Track</b> column.</li>
  </ul></li>
</ol>

<h2>Who is who</h2>
<ul>
<li><b>Subject</b> = the black mouse with the <b>orange headstage</b> on its head and the clear tube. It is in the cage at the start.</li>
<li><b>RI1 videos: Juvenile</b> = the smaller black mouse with no headstage, put in at the start of the video.</li>
<li><b>RI2 video: Agent</b> = the <b>white</b> mouse. Swaps are easy to see here: the Subject color must always be on the black mouse.</li>
<li>Don't rename tracks. If you see a third track name (like <i>track_2</i>), tell Sequioa.</li>
</ul>

<h2>Correct the IDs</h2>
<ol>
<li><b>The hand.</b> At the start the experimenter's <b>hand</b> (blue glove) puts the intruder in. Click any skeleton on the hand and press <kbd>Delete</kbd>, frame by frame, until the hand is gone.</li>
<li><b>Go through the whole video</b> with <kbd>Ctrl</kbd>+<kbd>→</kbd> (a few frames at a time), watching the colors. Slow down with <kbd>→</kbd> (1 frame) whenever the mice touch, fight or cross: those are the swap spots (the dips in the Min Centroid Proximity line).</li>
<li><b>Fixing a swap.</b> Go to the <b>first frame where the names are wrong</b> and press <kbd>Ctrl</kbd>+<kbd>T</kbd> (Tracks → Transpose Instance Tracks). This swaps the two names <b>from this frame to the end of the video</b>.
  <ul>
  <li>If the names flip back wrong again later, press <kbd>Ctrl</kbd>+<kbd>T</kbd> again at the first wrong frame there.</li>
  <li>A swap that lasts only a few frames needs <b>two</b> presses: one at its first wrong frame, one at the first frame after it that is wrong again.</li>
  <li>Always go forward in time. Fixing an earlier frame after a later one undoes the later fix.</li>
  </ul></li>
<li><b>Other problems.</b> Two skeletons on one mouse, or a skeleton on nothing: click the wrong one and press <kbd>Delete</kbd>. A mouse missing for a few frames: leave it. A body point slightly off: leave it.</li>
<li>Save often with <kbd>Ctrl</kbd>+<kbd>S</kbd>.</li>
</ol>
<p class="note">These videos are 30 frames per second and about 10 minutes long (~18,000 frames), so each takes longer than a cohort 2 video. It's fine to do one video over several sittings: Save, close, and next time open your <code>corrected/…idcorrected.slp</code> copy (not the one in <code>to_correct/</code>).</p>

<h2>When you finish a video</h2>
<ol>
<li><kbd>Ctrl</kbd>+<kbd>S</kbd> one last time.</li>
<li><b>File → Export Analysis HDF5… → Current video</b>, saved in <code>corrected/</code> with the same name ending in <code>.1.analysis.h5</code> (e.g. <code>{ex}.1.analysis.h5</code>).</li>
<li>In the <b>Notion</b> SLEAP ID Correction table, open your video's row, set <b>Tracking Status</b> to <i>Uploaded to Dropbox</i>, and write in the <b>Notes</b> column: your name, the date, how long it took, and anything odd (frame numbers you weren't sure about).</li>
<li>Tell Sequioa it's done.</li>
</ol>
<p class="note">If you're unsure about anything, ask Sequioa.</p>
</main></body></html>
'''
OUT.mkdir(parents=True, exist_ok=True); (OUT/'to_correct').mkdir(exist_ok=True); (OUT/'corrected').mkdir(exist_ok=True)
(OUT/'START_HERE.html').write_text(page, encoding='utf-8')
print(OUT/'START_HERE.html')
for r in rows: print(f'{r[0]:9s} {r[1]:36s} {r[4]}')
