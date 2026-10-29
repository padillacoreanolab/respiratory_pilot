# Retrack valence RI videos whose SLEAP was run on re-encoded copies (frames did not match Trodes).
# Same models and tracker settings as Tanish's original runs (read from his .predictions.slp provenance):
#   RI1: 250707_125107.centroid.n=701 + 250707_130232.centered_instance.n=701, tracking.match hungarian
#   RI2: 250626_150000.centroid.n=787.c57v.cd1 + 250626_151314.centered_instance.n=787.C57v.CD1, tracking.match greedy
#   both: max_instances 2, peak_threshold 0.2, simplemaxtracks, max_tracks 2, similarity instance, track_window 5
# Input: original .1.mp4 (frame count = .videoTimeStamps). Output: <video>.tracked.slp in to_correct/.
# Usage: powershell -File scripts/retrack_valence_sleap.ps1
$e = "C:\Users\sjs93\anaconda3\envs\sleap"
$env:PATH = "$e;$e\Library\mingw-w64\bin;$e\Library\usr\bin;$e\Library\bin;$e\Scripts;$env:PATH"
$M = "C:\Users\sjs93\UFL Dropbox\Sequioa Smith\Padilla-Coreano Lab\2026\tanish\models"
$R = "C:\Users\sjs93\UFL Dropbox\Sequioa Smith\Padilla-Coreano Lab\2025\R34_valence_SLEAP_retrack"
$jobs = @(
  @{v = "RI1_s4_7_p5_2_nRB3_20250621_150707.1.mp4"; c = "250707_125107.centroid.n=701"; i = "250707_130232.centered_instance.n=701"; m = "hungarian"},
  @{v = "RI1_s4_8_p5_1_nRB3_20250621_165214.1.mp4"; c = "250707_125107.centroid.n=701"; i = "250707_130232.centered_instance.n=701"; m = "hungarian"},
  @{v = "RI2_s2_3_p5_3_nRB3_20250622_110216.1.mp4"; c = "250626_150000.centroid.n=787.c57v.cd1"; i = "250626_151314.centered_instance.n=787.C57v.CD1"; m = "greedy"}
)
foreach ($j in $jobs) {
  $out = Join-Path $R ("to_correct\" + ($j.v -replace '\.mp4$', '.tracked.slp'))
  if (Test-Path $out) { "skip (exists) $out"; continue }
  & "$e\Scripts\sleap-track.exe" (Join-Path $R "videos\$($j.v)") -m (Join-Path $M $j.c) -m (Join-Path $M $j.i) `
    --max_instances 2 --peak_threshold 0.2 --batch_size 4 `
    --tracking.tracker simplemaxtracks --tracking.max_tracking 1 --tracking.max_tracks 2 `
    --tracking.similarity instance --tracking.match $j.m --tracking.track_window 5 `
    --verbosity rich -o $out
}
