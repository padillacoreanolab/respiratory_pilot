const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = process.argv[2];
const F = "Calibri";
const INK = "1F2937", MUTED = "4B5563", ACCENT = "1D4E89", TINT = "EAF0F7", RULE = "C9D1DB", WHITE = "FFFFFF", NAVY = "16304F";
const Q = { "Good": "DDEFE3", "Review": "FCEFD5", "Bad": "F8DCD9" };
// SSLD encoding
const RESP = "1D4E89", RESP_T = "DCE7F5", BEH = "B45A1C", BEH_T = "FBE6D6", SH = "6B4FA0", SH_T = "ECE5F6", OUTC = "4B5563", OUT_T = "EEF0F3", OPEN = "B45309", OPEN_T = "FEF3C7";
const DOCS = {
  sync: "https://docs.spikegadgets.com/en/latest/basic/UntetheredDataSynchronization.html",
  camera: "https://docs.spikegadgets.com/en/latest/basic/CameraModule.html",
  export: "https://docs.spikegadgets.com/en/latest/basic/ExportFunctions.html",
  boris: "https://www.boris.unito.it/",
  sleap: "https://sleap.ai/",
};
const img = (n) => path.join(__dirname, n);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Cohort 2 cagemate dataset";

// ---------- helpers ----------
function title(s, t, sub) {
  s.background = { color: WHITE };
  s.addText(t, { x: 0.5, y: 0.3, w: 12.33, h: 0.75, fontFace: F, fontSize: 32, bold: true, color: INK, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.5, y: 1.0, w: 12.33, h: 0.42, fontFace: F, fontSize: 17, color: MUTED, margin: 0, isTextBox: true });
}
const T = (text, o = {}) => ({ text, options: o });
function txt(s, runs, x, y, w, h, o = {}) {
  s.addText(typeof runs === "string" ? [T(runs)] : runs, Object.assign({ x, y, w, h, fontFace: F, fontSize: 15, color: INK, margin: 0, valign: "top", isTextBox: true }, o));
}
function rbox(s, x, y, w, h, fill, line, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line || fill, width: o.lw || 1, dashType: o.dash ? "dash" : "solid" }, rectRadius: 0.07 });
}
function box(s, x, y, w, h, head, sub, fill, headColor, o = {}) {
  rbox(s, x, y, w, h, fill, o.line, o);
  const runs = [T(head, { bold: true, fontSize: o.hs || 16, color: headColor || INK, breakLine: !!sub })];
  if (sub) runs.push(T(sub, { fontSize: o.ss || 13, color: MUTED }));
  txt(s, runs, x + 0.1, y, w - 0.2, h, { align: o.align || "center", valign: "middle" });
}
function line(s, x1, y1, x2, y2, o = {}) {
  const ln = { color: o.color || MUTED, width: o.width || 2, dashType: o.dash ? "dash" : "solid" };
  if (!o.noArrow) ln.endArrowType = "triangle";
  if (o.both) ln.beginArrowType = "triangle";
  const opt = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: ln };
  if (x2 < x1) opt.flipH = true;
  if (y2 < y1) opt.flipV = true;
  s.addShape(pres.shapes.LINE, opt);
}
function pill(s, x, y, label, fill, color) {
  rbox(s, x, y, 0.95, 0.3, fill, color);
  txt(s, label, x, y, 0.95, 0.3, { fontSize: 11, bold: true, color, align: "center", valign: "middle" });
}
const hdr = (cells) => cells.map(t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: ACCENT } } }));
const link = (label, url, o = {}) => T(label, Object.assign({ hyperlink: { url }, color: ACCENT, underline: { style: "sng" } }, o));

// ================= 0. Section divider (user's slide) =================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  txt(s, "H5s Exported", 0.8, 2.55, 11.7, 1.1, { fontSize: 54, bold: true, color: WHITE, valign: "bottom" });
  txt(s, "Trodes → H5 Export Errors", 0.8, 3.75, 11.7, 0.7, { fontSize: 28, color: "C7D6EA" });
}

// ================= 1. Dataset overview =================
{
  const s = pres.addSlide();
  title(s, "Cohort 2 cagemate dataset: what is this?", "BLA ChR resp cohort 2 · home-cage cagemate sessions · recorded Jul 29, 2026");
  // experimental unit diagram
  box(s, 0.5, 1.75, 3.3, 1.5, "Recorded mouse", "respiration sensor + video", RESP_T, RESP, { hs: 20, ss: 15 });
  box(s, 5.0, 1.75, 3.3, 1.5, "Cagemate", "home-cage partner · video only", BEH_T, BEH, { hs: 20, ss: 15 });
  line(s, 3.85, 2.5, 4.95, 2.5, { both: true, color: INK });
  txt(s, "interact", 3.8, 2.05, 1.2, 0.35, { fontSize: 13, italic: true, color: MUTED, align: "center" });
  txt(s, [T("One recording = one recorded mouse + one cagemate from its own cage, ~10–12 min. ", { bold: true }),
          T("Each mouse has a rank: D = dominant, S = subordinate.")], 0.5, 3.4, 7.8, 0.8, { fontSize: 15 });
  // filename anatomy
  txt(s, "Reading a file name", 0.5, 4.35, 7.8, 0.4, { fontSize: 18, bold: true, color: ACCENT });
  const parts = [["13_2", "recorded mouse\n(cage 13, #2)", RESP_T], ["d", "its rank:\ndominant", RESP_T], ["CM", "cagemate\nsession", "E5E7EB"],
                 ["13_1", "cagemate\n(cage 13, #1)", BEH_T], ["s", "its rank:\nsubordinate", BEH_T], ["20260729_111038", "date + start\ntime (11:10)", "E5E7EB"]];
  const W = [1.15, 0.95, 1.0, 1.15, 1.1, 2.15]; let x = 0.5;
  parts.forEach(([k, d, fill], i) => {
    rbox(s, x, 4.85, W[i], 0.55, fill);
    txt(s, k, x, 4.85, W[i], 0.55, { fontSize: 18, bold: true, align: "center", valign: "middle" });
    txt(s, d, x - 0.02, 5.47, W[i] + 0.04, 0.7, { fontSize: 12, color: MUTED, align: "center" });
    x += W[i] + 0.1;
  });
  txt(s, "One file uses a different order: 13_3_CM_s_13_2_d.", 0.5, 6.35, 7.8, 0.4, { fontSize: 13, italic: true, color: MUTED });
  // key numbers
  const stats = [["11", "recordings"], ["7", "cagemate pairs · cages 10, 11, 13, 14"], ["3 of 7", "pairs recorded from both mice"], ["5 of 11", "sessions have BORIS scoring"]];
  stats.forEach(([b, sm], i) => {
    txt(s, [T(b, { fontSize: 38, bold: true, color: ACCENT, breakLine: true }), T(sm, { fontSize: 15, color: MUTED })], 9.2, 1.7 + i * 1.3, 3.7, 1.2);
  });
  s.addNotes("Experimental unit: one recorded (implanted) mouse interacting with one home-cage cagemate. File names give recorded mouse, its rank, CM, the cagemate and its rank, then date and start time. I'm assuming the first ID is the recorded mouse (consistent across files). Rank labels come from the file names (d/s).");
}

// ================= 2. Pairings + data =================
{
  const s = pres.addSlide();
  title(s, "Cagemate pairings and what data exists", "Shaded rows: both mice of the pair were recorded");
  const BOTH = "F4F1FA";
  const q = (t) => T(t === "Review" ? "Somewhat bad" : t, { fill: { color: Q[t] }, bold: true });
  const r = (cells, both) => cells.map(c => typeof c === "string" ? T(c, both ? { fill: { color: BOTH } } : {}) : c);
  const rows = [
    [T("10", { rowspan: 2 }), ...r(["10.2", "D", "10.1", "S", "11:44"]), q("Bad"), T("")],
    [...r(["10.3", "S", "10.2", "D", "17:41"]), q("Good"), T("")],
    [T("11", { rowspan: 3 }), ...r(["11.2", "S", "11.4", "D", "17:23"]), q("Review"), T("")],
    [...r(["11.3", "S", "11.4", "D", "12:20"], 1), q("Bad"), T("", { fill: { color: BOTH } })],
    [...r(["11.4", "D", "11.3", "S", "12:50"], 1), q("Good"), T("", { fill: { color: BOTH } })],
    [T("13", { rowspan: 4 }), ...r(["13.1", "S", "13.2", "D", "17:01"], 1), q("Good"), T("", { fill: { color: BOTH } })],
    [...r(["13.2", "D", "13.1", "S", "11:10"], 1), q("Good"), T("Yes", { fill: { color: BOTH }, bold: true })],
    [...r(["13.2", "D", "13.1", "S", "18:13"], 1), q("Good"), T("Yes", { fill: { color: BOTH }, bold: true })],
    [...r(["13.3", "S", "13.2", "D", "17:56"]), q("Good"), T("Yes", { bold: true })],
    [T("14", { rowspan: 2 }), ...r(["14.1", "D", "14.3", "S", "12:03"], 1), q("Bad"), T("Yes", { fill: { color: BOTH }, bold: true })],
    [...r(["14.3", "S", "14.1", "D", "11:25"], 1), q("Good"), T("Yes", { fill: { color: BOTH }, bold: true })],
  ];
  s.addTable([hdr(["Cage", "Recorded", "Rank", "Cagemate", "Rank", "Time", "Respiration", "BORIS"]), ...rows],
    { x: 0.5, y: 1.55, w: 7.6, colW: [0.7, 1.05, 0.7, 1.15, 0.7, 0.85, 1.5, 0.95], fontFace: F, fontSize: 14, color: INK, align: "center", valign: "middle", border: { type: "solid", pt: 0.75, color: RULE }, rowH: 0.42, margin: [0.02, 0.04, 0.02, 0.04] });
  txt(s, "13.2 with 13.1 was recorded twice (11:10 and 18:13). D = dominant, S = subordinate.", 0.5, 6.7, 7.6, 0.4, { fontSize: 13, italic: true, color: MUTED });

  // data available
  txt(s, "Data available", 8.5, 1.5, 4.4, 0.4, { fontSize: 18, bold: true, color: ACCENT });
  const data = [
    [T("Respiration", { bold: true }), T("Breathing signal, all 11 (h5_outputs)")],
    [T("Video", { bold: true }), T("Home-cage interaction, all 11 (15 fps)")],
    [T("BORIS", { bold: true }), [link("Manual scoring", DOCS.boris, { fontSize: 13 }), T(", 5 of 11: facial / body / anogenital sniffing, digging, self-grooming", { fontSize: 13 })]],
    [T("SLEAP", { bold: true }), [link("Pose tracking", DOCS.sleap, { fontSize: 13 }), T(": not run yet (SLEAP_tracked empty)", { fontSize: 13 })]],
    [T("Trodes timestamps", { bold: true }), T("Clock tick per sample and frame: aligns everything")],
  ].map(([a, b]) => [a, Array.isArray(b) ? { text: b } : b]);
  s.addTable(data, { x: 8.5, y: 1.95, w: 4.35, colW: [1.35, 3.0], fontFace: F, fontSize: 13, color: INK, valign: "middle", border: { type: "solid", pt: 0.75, color: RULE }, rowH: 0.55, margin: [0.04, 0.06, 0.04, 0.06] });
  rbox(s, 8.5, 5.05, 4.35, 1.5, OPEN_T, OPEN);
  txt(s, [T("Missing recordings?", { bold: true, color: OPEN, fontSize: 15, breakLine: true }),
          T("The cohort 2 roster (RI1 / OFT folders) has 18 mice in cages 10–14. No cagemate recordings for cage 12 (12.2–12.4) or for 10.1, 10.4, 13.4, 14.2, 14.4. Check whether these were planned.", { fontSize: 13 })],
    8.65, 5.12, 4.05, 1.85);
  s.addNotes("BORIS behaviors read from the 5 CSVs: facial sniffing, body sniffing, anogenital sniffing, digging, self-grooming; all scored as STATE events (start/stop) for both the subject and the social agent. H5 files: UF Dropbox …\\2026\\BLA_ChR_resp_cohort2\\BLA_ChR_resp_CM_hc\\h5_outputs\\ (made with src/file_conversion/ekg_resp_to_h5.py in biopipeline-env). SLEAP: the cohort 2 SLEAP_tracked folder is empty and the model used for the older cohorts hasn't been located (likely on the lab PC). Missing recordings: roster taken from mouse IDs in the cohort 2 RI1 and OFT h5 file names.");
}

// ================= 3. Good vs bad =================
{
  const s = pres.addSlide();
  title(s, "Good vs. bad respiration", "Processed signal (400 Hz, 0.1–20 Hz band-pass) · first three panels on the same scale");
  s.addImage({ path: img("qc_teach.png"), x: 0.4, y: 1.5, w: 6.75, h: 6.75 * 1160 / 1400 });
  rbox(s, 7.5, 1.55, 5.35, 1.45, Q["Good"]);
  txt(s, [T("Good looks like", { bold: true, fontSize: 18, color: "2E7D4F", breakLine: true }),
          T("Clear, repeating breaths", { bullet: true, breakLine: true }), T("Strong, visible rhythm (~10 Hz here)", { bullet: true, breakLine: true }), T("Fairly stable amplitude", { bullet: true })],
    7.7, 1.62, 5.0, 1.5, { fontSize: 16 });
  rbox(s, 7.5, 3.15, 5.35, 1.45, Q["Bad"]);
  txt(s, [T("Bad looks like", { bold: true, fontSize: 18, color: "B42318", breakLine: true }),
          T("Very weak, almost flat breathing", { bullet: true, breakLine: true }), T("Breaths buried in slow drift", { bullet: true, breakLine: true }), T("Large movement / contact artifacts", { bullet: true })],
    7.7, 3.22, 5.0, 1.4, { fontSize: 16 });
  // QC reference
  rbox(s, 7.5, 4.75, 5.35, 1.75, "F3F4F6", RULE);
  txt(s, [T("QC reference (10-s windows)", { bold: true, fontSize: 14, color: ACCENT, breakLine: true }),
          T("Signal loss: ", { bold: true }), T("window SD < 25% of cohort median", { breakLine: true }),
          T("Artifact: ", { bold: true }), T("peak > 4× cohort median 99th pct", { breakLine: true }),
          T("Weak rhythm: ", { bold: true }), T("< 60% of power in 1–15 Hz", { breakLine: true }),
          T("Session: ", { bold: true }), T("Good ≥ 85% of windows pass · Somewhat bad (review) 60–85% · Bad < 60%")],
    7.65, 4.82, 5.05, 1.65, { fontSize: 13 });
  s.addNotes("Processing: preprocess_resp_to_target_rate (low-pass, downsample to 400 Hz, 0.1–20 Hz Butterworth band-pass). Thresholds are relative to the whole cohort (median window SD 323 a.u., median 99th percentile 909 a.u.) so one bad session can't lower its own bar. The artifact example comes from 13_1_s, which is a Good session overall: artifacts happen in good recordings too; the QC flags those windows. Thresholds are proposed, not yet approved for exclusions.");
}

// ================= 4. Quality across cohort =================
{
  const s = pres.addSlide();
  title(s, "7 Good  |  1 Somewhat bad  |  3 Bad", "Trust: Good sessions · Use with caution: 11_2_s, 14_3_s · Check before use: 10_2_d, 11_3_s, 14_1_d");
  s.addImage({ path: img("traces_qc_annot.png"), x: 0.65, y: 1.5, w: 12.0, h: 12.0 * 1280 / 2600 });
  s.addNotes("Full-session processed respiration (400 Hz), ordered Good → Bad, same amplitude scale in every panel; shaded = flagged 10-s window. Amplitude is ADC counts (a.u.), not volts. Large artifacts are clipped at the panel edge.");
}

// ================= 5. Alignment =================
{
  const s = pres.addSlide();
  title(s, "How behavior, video and respiration line up", "Everything is matched through the Trodes clock (20,000 ticks per second)");
  const Y = 1.75, H = 1.35;
  box(s, 0.5, Y, 3.2, H, "BORIS event / video frame", "e.g. frame 300 of the mp4", BEH_T, BEH, { hs: 18, ss: 14 });
  box(s, 4.85, Y, 3.4, H, "Trodes timestamp", "the clock tick stamped on that frame", "E5E7EB", INK, { hs: 18, ss: 14 });
  box(s, 9.4, Y, 3.45, H, "Respiration sample", "the sample with the same tick", RESP_T, RESP, { hs: 18, ss: 14 });
  line(s, 3.75, Y + H / 2, 4.8, Y + H / 2, { color: INK, width: 2.5 });
  line(s, 8.3, Y + H / 2, 9.35, Y + H / 2, { color: INK, width: 2.5 });
  txt(s, [T("In each h5: ", { bold: true }), T("/video/frame_to_resp_index[frame] does this lookup; video_time_to_resp_time() converts BORIS seconds.")],
    0.5, 3.25, 12.33, 0.4, { fontSize: 14, color: MUTED });
  // cohort cards
  rbox(s, 0.5, 3.9, 6.0, 1.85, TINT);
  txt(s, [T("Cohort 2: tethered", { bold: true, fontSize: 18, color: ACCENT, breakLine: true }),
          T("Respiration and camera frames are stamped by the same clock.", { bullet: true, breakLine: true }),
          T("No wireless headstage, so no merge step.", { bullet: true, breakLine: true }),
          T("Video and respiration start within 0.07 s of each other.", { bullet: true })], 0.7, 4.0, 5.6, 2.2, { fontSize: 15 });
  rbox(s, 6.85, 3.9, 6.0, 1.85, "F3E8D6");
  txt(s, [T("Aim1: wireless", { bold: true, fontSize: 18, color: BEH, breakLine: true }),
          T("The headstage keeps its own clock; files are merged afterwards.", { bullet: true, breakLine: true }),
          T("Merging adds copied samples and can cut the start of respiration.", { bullet: true, breakLine: true }),
          T("Only timestamps line things up correctly.", { bullet: true })], 7.05, 4.0, 5.6, 2.2, { fontSize: 15 });
  txt(s, [T("So: ", { bold: true }), T("use timestamps, not sample number ÷ 20,000 and not mp4 seconds. "),
          T("Docs: "), link("Trodes camera module (video timestamps)", DOCS.camera)], 0.5, 6.0, 12.33, 0.5, { fontSize: 15 });
  s.addNotes("Respiration: sensor → ECU analog input 1 → digitized → MCU → Trodes, every sample stamped with the MCU tick. Camera: every frame is stamped with the MCU tick when it arrives (.videoTimeStamps). BORIS works in mp4 time, so a BORIS time is converted to a frame number (t × mp4 fps), then to that frame's tick, then to the respiration sample with that tick. Some mp4 files report a frame rate different from what the camera captured (e.g. 13_2_d 11:10 is encoded at 16 fps vs 15 captured) — going through the frame number handles that.");
}

// ================= 5b. Drift + offset explained =================
{
  const s = pres.addSlide();
  title(s, "The two timing problems, visually", "Old Aim1 (wireless, merged) files only · cohort 2 has neither");
  const RED = "B42318";
  // ---- left: drift ----
  txt(s, "1  Drift: copied samples stretch the time axis", 0.5, 1.55, 6.1, 0.4, { fontSize: 17, bold: true, color: ACCENT });
  const ticks = ["1", "2", "3", "4", "4", "4", "5", "6", "7", "8", "8", "9"];
  const dup = new Set([4, 5, 10]);
  ticks.forEach((t2, i) => {
    const x = 0.5 + i * 0.5;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.05, w: 0.44, h: 0.44, fill: { color: dup.has(i) ? "F8DCD9" : RESP_T }, line: { color: dup.has(i) ? RED : RESP, width: 1 } });
    txt(s, t2, x, 2.05, 0.44, 0.44, { fontSize: 13, bold: true, color: dup.has(i) ? RED : RESP, align: "center", valign: "middle" });
  });
  txt(s, [T("Trodes tick of each sample. ", { color: MUTED }), T("Red = copy added by the merge", { bold: true, color: RED }), T(" (same tick, same value).", { color: MUTED })],
    0.5, 2.55, 6.1, 0.35, { fontSize: 12 });
  txt(s, "Old code: time = sample number ÷ 20,000, so each copy adds fake time: respiration is read too early (~27,000 copies per session):",
    0.5, 2.9, 6.1, 0.6, { fontSize: 14 });
  s.addImage({ path: img("drift2.png"), x: 0.6, y: 3.5, w: 5.0, h: 5.0 * 720 / 1240 });
  // ---- right: offset ----
  txt(s, "2  Offset: the merge cuts the start of respiration", 6.95, 1.55, 5.9, 0.4, { fontSize: 17, bold: true, color: ACCENT });
  const x0 = 6.95, sc = 1.95; const X = (t2) => x0 + t2 * sc;
  [["Camera frames", 0, ACCENT], ["Respiration, unmerged", 0, "4B8B5E"], ["Respiration, merged (old h5s)", 1.094, BEH]].forEach(([lab, t0, col], i) => {
    const y = 2.05 + i * 0.85;
    txt(s, lab, x0, y, 5.85, 0.3, { fontSize: 13, bold: true });
    if (t0 > 0) {
      s.addShape(pres.shapes.RECTANGLE, { x: X(0), y: y + 0.32, w: t0 * sc, h: 0.32, fill: { color: "E5E7EB" }, line: { color: "9CA3AF", dashType: "dash", width: 1 } });
      txt(s, "cut by merge", X(0), y + 0.32, t0 * sc, 0.32, { fontSize: 11, italic: true, color: MUTED, align: "center", valign: "middle" });
    }
    s.addShape(pres.shapes.RECTANGLE, { x: X(t0), y: y + 0.32, w: X(3) - X(t0), h: 0.32, fill: { color: col }, line: { color: col } });
  });
  line(s, X(0), 4.75, X(3), 4.75, { noArrow: true, color: INK, width: 1.25 });
  [0, 1, 2, 3].forEach(t2 => {
    line(s, X(t2), 4.75, X(t2), 4.85, { noArrow: true, color: INK, width: 1.25 });
    txt(s, `${t2} s`, X(t2) - 0.3, 4.88, 0.6, 0.3, { fontSize: 12, color: MUTED, align: "center" });
  });
  txt(s, "1.094 s", X(0), 4.42, 1.094 * sc, 0.3, { fontSize: 13, bold: true, color: BEH, align: "center" });
  txt(s, [T("Example: ", { bold: true }), T("a BORIS event at 2.0 s of video. Old code read merged respiration at 2.0 s, which is really 3.09 s: "), T("1.09 s late.", { bold: true, color: RED })],
    6.95, 5.3, 5.9, 0.8, { fontSize: 14 });
  txt(s, "Aim1 offsets ranged 0–1.1 s across 25 sessions.", 6.95, 6.1, 5.9, 0.35, { fontSize: 13, color: MUTED });
  // ---- bottom ----
  rbox(s, 0.5, 6.6, 12.35, 0.6, Q["Good"]);
  txt(s, [T("Opposite directions: ", { bold: true }), T("offset → read too late, copies → read too early. "), T("Fix for both: ", { bold: true, color: "2E7D4F" }), T("remove copies; match via Trodes timestamps.")], 0.7, 6.6, 12.0, 0.6, { fontSize: 14, valign: "middle" });
  s.addNotes("Drift: during the wireless merge, Trodes keeps every headstage packet and pads the respiration (MCU/ECU) stream with copies of the previous sample to keep the two in step (documented SpikeGadgets behavior). In Aim1 this happened ~450 times per 10 s, regularly every 0.1 s: a steady clock-rate mismatch, far above the 'few dozen per 10 s' Trodes calls normal. Counting copies as time made the old time axis run late, linearly, reaching 1.37 s at the end of CM_s1_1. Verified: removing the copies reproduces the unmerged recording exactly. Offset: in CM_s2_4_sub2_3_1d, camera and unmerged respiration start on the same tick (975,650 vs 975,651) but merged respiration starts 1.094 s later. BORIS times are video times, so events were matched to respiration up to ~1.1 s late. The two errors push in opposite directions: the start offset makes the old code read respiration too late (constant), the copies make it read respiration too early (growing through the session). Measured on real BORIS events in 6 Aim1 baseline sessions: net error from +0.4 s early in a session to -1.3 s by the end.");
}

// ================= 5c. Real BORIS misalignment =================
{
  const s = pres.addSlide();
  title(s, "Real BORIS events: how far off the old alignment was", "Aim1 baseline cagemate sessions · subject sniff onsets · old pipeline vs. corrected");
  txt(s, "Error for every BORIS event (old pipeline)", 0.5, 1.5, 6.4, 0.35, { fontSize: 16, bold: true, color: ACCENT });
  s.addImage({ path: img("misalign_events.png"), x: 0.5, y: 1.9, w: 6.4, h: 6.4 * 880 / 1440 });
  txt(s, "Each line is one session: starts at the merge offset (late), then the copies pull it early. Net −1.3 s to +0.4 s.",
    0.5, 5.85, 6.4, 0.6, { fontSize: 13, color: MUTED });
  txt(s, "Physiology check: breathing rate at sniff onset", 7.15, 1.5, 5.7, 0.35, { fontSize: 16, bold: true, color: ACCENT });
  s.addImage({ path: img("misalign_eta.png"), x: 7.15, y: 1.9, w: 5.7, h: 5.7 * 880 / 1240 });
  txt(s, "Corrected: the breathing-rate peak sits at the BORIS onset. Old: lower and less sharply aligned. A modest effect, because the two errors partly cancel.",
    7.15, 5.95, 5.7, 0.6, { fontSize: 13, color: MUTED });
  rbox(s, 0.5, 6.6, 12.35, 0.6, Q["Bad"]);
  txt(s, [T("CM_s4_7_d4_8 (baseline): ", { bold: true, color: "B42318" }), T("BORIS was scored on a 23 fps mp4 (camera captured 30 fps), so old alignment was off by 45–125 s. Every old behavior-locked result for this session is wrong.")],
    0.7, 6.6, 12.0, 0.6, { fontSize: 14, valign: "middle" });
  s.addNotes("Per-event error = Trodes tick of the respiration sample the old code used (BORIS Start (s) × 20,000 into the merged array) minus the Trodes tick of the BORIS frame (Image index start → .videoTimeStamps). Positive = respiration read too late. Events: subject-initiated facial/body/anogenital sniffing from the Aim1 baseline cagemate BORIS exports (cm_boris/baseline csv; CM_s4_8_sub4_7_20250623.csv excluded as in build_interaction_boris_catalog; CM_s3_6 file is empty). Breathing rate: preprocess_resp_to_target_rate to 400 Hz, inhalation peaks, instantaneous rate, averaged ±6 s around onsets (mean ± SEM, n = 185 each). CM_s3_5_d3_6_20250623_160001 has a truncated respiration export, so its events stop at ~5.7 min.");
}

// ================= 6. What was wrong + rules =================
{
  const s = pres.addSlide();
  title(s, "What was wrong, and the rules going forward", "Both problems affected the old Aim1 (wireless) h5s; cohort 2 is not affected");
  const card = (y, head, body) => {
    rbox(s, 0.5, y, 6.4, 1.3, "F8DCD9");
    txt(s, [T(head, { bold: true, fontSize: 16, color: "B42318", breakLine: true }), ...body], 0.7, y + 0.08, 6.0, 1.45, { fontSize: 14 });
  };
  card(1.55, "Problem 1: duplicate samples → drifting time", [T("Merging padded respiration with ~27,000 copied samples per session. Treating sample number as time matched BORIS events to respiration up to ~1.3 s too early by the end. ", { breakLine: true }), link("Trodes untethered sync docs", DOCS.sync, { fontSize: 13 })]);
  card(3.0, "Problem 2: respiration started after video", [T("The merge cut 0–1.1 s from the start of respiration, so BORIS events were matched to respiration up to 1.1 s late.")]);
  rbox(s, 0.5, 4.45, 6.4, 1.3, Q["Good"]);
  txt(s, [T("Fix", { bold: true, fontSize: 16, color: "2E7D4F", breakLine: true }),
          T("Remove the copies (result matches the unmerged recording exactly), keep Trodes timestamps, and map video frame → Trodes tick → respiration sample.")], 0.7, 4.53, 6.0, 1.2, { fontSize: 14 });
  // rules
  rbox(s, 7.2, 1.55, 5.65, 4.2, TINT, ACCENT, { lw: 1.5 });
  const rules = ["Use Trodes timestamps for alignment.", "Don't compute time as sample index ÷ 20,000.", "Don't trust mp4 fps for precise behavioral alignment.",
                 "Respiration values are raw ADC counts (a.u.), not volts.", "Cohort 2 is tethered: no Aim1 merge correction needed.",
                 "Keep /timestamps and /video mappings in every h5.", "Check respiration QC before downstream analysis."];
  txt(s, [T("Rules to remember", { bold: true, fontSize: 20, color: ACCENT, breakLine: true }),
          ...rules.map((r, i) => T(r, { bullet: { type: "number" }, breakLine: i < rules.length - 1, paraSpaceBefore: 6 }))],
    7.45, 1.7, 5.2, 4.0, { fontSize: 16 });
  txt(s, [T("Trodes docs: ", { bold: true }), link("untethered data sync", DOCS.sync), T(" · "), link("camera module", DOCS.camera), T(" · "), link("export functions", DOCS.export)], 0.5, 6.1, 12.35, 0.45, { fontSize: 14 });
  s.addNotes("Evidence: for Aim1 CM_s1_1_d1_2 the unmerged .rec export has 12,224,931 samples; the merged file has 12,252,312; removing the 27,381 repeats reproduces the unmerged timestamps and values exactly. The copies arrive every 0.1 s (~450 per 10 s), far above the 'few dozen per 10 s' Trodes calls normal — a steady clock-rate mismatch between headstage and MCU; worth asking SpikeGadgets. Start offset: for CM_s2_4_sub2_3_1d the camera and unmerged respiration start on the same tick (975,650 vs 975,651) but merged respiration starts 1.094 s later. Old Aim1 h5s still have both problems and need regenerating before BORIS-locked analyses. Other old issues fixed: sampling rate now read from the file header, units relabeled, tethered .rec files supported, IMU and video timing saved. Export docs: " + DOCS.export);
}

// ================= 7. SSLD concept =================
{
  const s = pres.addSlide();
  title(s, "Next step: what SSLD would do with our data", "Switching Shared Latent Dynamics · planned, not yet tested");
  // inputs
  box(s, 0.5, 1.75, 2.75, 1.55, "Respiration", "physiological time series\nanalyzed at 50 Hz", RESP_T, RESP, { hs: 18, ss: 13 });
  box(s, 0.5, 4.45, 2.75, 1.75, "SLEAP / movement", "tracked body nodes, 15 or 20 fps\n→ distance, velocity, orientation,\napproach / withdrawal", BEH_T, BEH, { hs: 18, ss: 12 });
  // encoders
  box(s, 3.75, 1.95, 1.6, 1.15, "Encoder", null, RESP_T, RESP, { hs: 15 });
  box(s, 3.75, 4.75, 1.6, 1.15, "Encoder", null, BEH_T, BEH, { hs: 15 });
  line(s, 3.27, 2.52, 3.72, 2.52, { color: RESP });
  line(s, 3.27, 5.32, 3.72, 5.32, { color: BEH });
  // latents
  box(s, 5.95, 1.6, 2.8, 1.1, "Respiration-private", "mainly in respiration", RESP_T, RESP, { hs: 15, ss: 12, line: RESP });
  box(s, 5.95, 3.05, 2.8, 1.75, "Shared latent", "present in both", SH_T, SH, { hs: 17, ss: 12, line: SH, lw: 2 });
  box(s, 5.95, 5.15, 2.8, 1.1, "Behavior-private", "mainly in visible movement", BEH_T, BEH, { hs: 15, ss: 12, line: BEH });
  line(s, 5.37, 2.4, 5.92, 2.15, { color: RESP }); line(s, 5.37, 2.75, 5.92, 3.55, { color: RESP });
  line(s, 5.37, 5.55, 5.92, 5.7, { color: BEH }); line(s, 5.37, 5.15, 5.92, 4.3, { color: BEH });
  // switching dynamics strip inside shared
  const segs = [["A", 0.7], ["B", 0.5], ["A", 0.45], ["C", 0.75]]; let sx = 6.15;
  const shades = { A: "C9B8E8", B: "A68BD6", C: "8466C2" };
  segs.forEach(([k, w]) => { s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 4.2, w, h: 0.32, fill: { color: shades[k] }, line: { color: WHITE, width: 1 } }); txt(s, k, sx, 4.2, w, 0.32, { fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle" }); sx += w; });
  txt(s, "switching dynamics: regimes over time", 5.95, 4.55, 2.8, 0.25, { fontSize: 11, italic: true, color: SH, align: "center" });
  // representation
  box(s, 9.2, 1.6, 1.65, 4.65, "Learned representation", "private + shared latents, regime sequence", "F3F4F6", INK, { hs: 15, ss: 12, line: RULE });
  [2.15, 3.9, 5.7].forEach(y => line(s, 8.78, y, 9.22, y, { color: INK }));
  // outcome
  line(s, 10.88, 3.92, 11.33, 3.92, { color: OUTC, dash: true });
  box(s, 11.38, 2.85, 1.47, 2.15, "Test", "Rank\n(D vs S)\nValence\n(+ vs −)", OUT_T, OUTC, { hs: 15, ss: 12, line: OUTC, dash: true });
  txt(s, "downstream test, not an input", 11.38, 5.1, 1.47, 0.6, { fontSize: 11, italic: true, color: OUTC, align: "center" });
  txt(s, [T("SSLD lets us ask whether socially relevant information is carried uniquely by respiration, uniquely by behavior, or by their shared dynamics.", { bold: true })],
    0.5, 6.5, 10.0, 0.7, { fontSize: 15 });
  txt(s, [T("Paper: "), T("[add SSLD paper link]", { color: OPEN, bold: true })], 10.6, 6.6, 2.3, 0.4, { fontSize: 12, align: "right" });
  s.addNotes("Two measured inputs, each with its own encoder. The model splits what it learns into a respiration-private part, a behavior-private part, and a shared part present in both. The shared part can switch between dynamical regimes over time; these may correspond to behavioral/social states, but that is something to test, not assume. Rank and valence are never inputs: we only test afterwards whether the learned representation carries them. Respiration: downsampled from 400 Hz to 50 Hz. SLEAP: 15 or 20 fps depending on the recording; cohort 2 has no SLEAP output yet.");
}

// ================= 8. Ablation =================
{
  const s = pres.addSlide();
  title(s, "How much does respiration actually contribute?", "Does respiration carry social information beyond what is already visible in behavior?");
  const dot = (on, col, f = {}) => T(on ? "●" : "", Object.assign({ color: col, fontSize: 18, bold: true }, f));
  const sets = [
    ["Respiration only", 1, 0, 0, "Physiological information"],
    ["Pose only", 0, 1, 0, "Visible movement information"],
    ["Behavioral states only", 0, 0, 1, "Interpretable behavior categories"],
    ["Respiration + pose", 1, 1, 0, "Does physiology add beyond movement?"],
    ["Respiration + states", 1, 0, 1, "Does physiology add beyond behavior?"],
    ["Pose + states", 0, 1, 1, "Behavior-only reference"],
    ["All three", 1, 1, 1, "Maximum multimodal information"],
  ];
  const key = new Set([3, 4]);
  const rows = sets.map(([n, r, p, b, why], i) => {
    const f = key.has(i) ? { fill: { color: SH_T } } : {};
    return [T(n, Object.assign({ bold: true, align: "left" }, f)), dot(r, RESP, f), dot(p, BEH, f), dot(b, "8A5A2B", f),
            T(why, Object.assign({ align: "left" }, f))];
  });
  s.addTable([hdr(["Input set", "Resp", "Pose", "States", "Purpose"]), ...rows],
    { x: 0.5, y: 1.6, w: 7.4, colW: [2.2, 0.7, 0.7, 0.8, 3.0], fontFace: F, fontSize: 14, color: INK, align: "center", valign: "middle", border: { type: "solid", pt: 0.75, color: RULE }, rowH: 0.5, margin: [0.03, 0.08, 0.03, 0.08] });
  txt(s, "Highlighted rows give the key comparison →", 0.5, 5.65, 7.4, 0.35, { fontSize: 13, italic: true, color: SH });
  // delta box
  rbox(s, 8.25, 1.6, 4.6, 1.95, SH_T, SH, { lw: 1.5 });
  txt(s, [T("Δ respiration contribution", { bold: true, fontSize: 18, color: SH, breakLine: true }),
          T("= Performance(Respiration + Behavior)", { fontSize: 16, breakLine: true }), T("   − Performance(Behavior alone)", { fontSize: 16, breakLine: true }),
          T("More informative than asking whether the full model predicts rank or valence: it isolates what respiration adds.", { fontSize: 13, color: MUTED })],
    8.45, 1.7, 4.25, 2.2);
  // targets
  txt(s, "Biological targets (tested separately)", 8.25, 3.8, 4.6, 0.35, { fontSize: 15, bold: true, color: OUTC });
  s.addTable([[T("Rank", { bold: true }), T("Dominant vs subordinate (per animal)")], [T("Valence", { bold: true }), T("Positive vs negative (separate experiment)")], [T("Cagemate", { bold: true }), T("Separate experiment / shared-behavior dataset")]],
    { x: 8.25, y: 4.2, w: 4.6, colW: [1.15, 3.45], fontFace: F, fontSize: 13, color: INK, valign: "middle", fill: { color: OUT_T }, border: { type: "solid", pt: 0.75, color: RULE }, rowH: 0.42, margin: [0.03, 0.08, 0.03, 0.08] });
  // open question
  rbox(s, 0.5, 6.05, 12.35, 0.98, OPEN_T, OPEN, { dash: true, lw: 1.25 });
  pill(s, 0.65, 6.15, "OPEN", WHITE, OPEN);
  txt(s, [T("Behavioral states are not yet labeled continuously. ", { bold: true }),
          T("BORIS marks social events (sniffing types, digging, self-grooming), not continuous states like approach or locomotion. We may hand-label a subset and/or try HMM / clustering to find candidate states, then check them against human labels; these methods are not guaranteed to recover meaningful states.")],
    1.75, 6.1, 11.0, 1.0, { fontSize: 13 });
  s.addNotes("The main goal is not maximum prediction accuracy but how important respiration is as an interpretable signal: how much unique information it contributes, whether it works alone, whether it needs pose or behavioral state, and which combinations are needed for statistically meaningful prediction. Rank labels are D/S per animal. Valence is positive vs negative only; cagemate is a separate experiment.");
}

// ================= 9. Temporal plan =================
{
  const s = pres.addSlide();
  title(s, "From continuous recording to session-level prediction", "Proposed training plan · window lengths and state labels still open");
  // timeline
  const st = [["Rest / exploration", 2.3, "E5E7EB"], ["Approach", 1.6, BEH_T], ["Social interaction", 2.6, SH_T], ["Withdrawal", 1.6, BEH_T], ["Non-social period", 2.6, "E5E7EB"]];
  let x = 0.5;
  st.forEach(([lab, w, f]) => { s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w, h: 0.55, fill: { color: f }, line: { color: WHITE, width: 2 } }); txt(s, lab, x, 1.6, w, 0.55, { fontSize: 14, bold: true, align: "center", valign: "middle" }); x += w; });
  txt(s, "schematic: these states are not yet labeled", x + 0.1, 1.62, 12.85 - x - 0.1, 0.55, { fontSize: 11, italic: true, color: MUTED, valign: "middle" });
  // pipeline
  const steps = ["Continuous recording", "Fixed-length multimodal windows", "SSLD", "Window / event summaries", "Rank or valence prediction"];
  const fills = ["F3F4F6", "F3F4F6", SH_T, "F3F4F6", OUT_T];
  const sw = 2.2, gap = 0.33; x = 0.5;
  steps.forEach((t2, i) => { box(s, x, 2.5, sw, 0.85, t2, null, fills[i], i === 2 ? SH : INK, { hs: 14, line: RULE }); if (i < steps.length - 1) line(s, x + sw + 0.03, 2.92, x + sw + gap - 0.03, 2.92, { color: INK }); x += sw + gap; });
  txt(s, "many windows from one mouse → state occupancy, transitions, latent summaries, respiration response → one prediction per mouse/session", 0.5, 3.4, 12.35, 0.35, { fontSize: 13, italic: true, color: MUTED });
  // event window
  txt(s, "Event-centered window (fixed length, ~20 s to start)", 0.5, 3.95, 6.6, 0.35, { fontSize: 15, bold: true, color: ACCENT });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.4, w: 1.5, h: 0.6, fill: { color: "E5E7EB" }, line: { color: WHITE, width: 2 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 2.0, y: 4.4, w: 2.5, h: 0.6, fill: { color: SH_T }, line: { color: WHITE, width: 2 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 4.5, y: 4.4, w: 2.6, h: 0.6, fill: { color: "E5E7EB" }, line: { color: WHITE, width: 2 } });
  txt(s, "5 s pre-event", 0.5, 4.4, 1.5, 0.6, { fontSize: 13, align: "center", valign: "middle" });
  txt(s, "interaction (variable)", 2.0, 4.4, 2.5, 0.6, { fontSize: 13, bold: true, align: "center", valign: "middle" });
  txt(s, "post-event context", 4.5, 4.4, 2.6, 0.6, { fontSize: 13, align: "center", valign: "middle" });
  txt(s, "▲ true onset", 1.62, 5.02, 1.2, 0.3, { fontSize: 11, color: SH }); txt(s, "▲ true offset", 4.12, 5.02, 1.2, 0.3, { fontSize: 11, color: SH });
  txt(s, [T("Same model also sees windows sampled from non-social periods.", { bullet: true, breakLine: true }),
          T("Whole 5–10 min recordings at 50 Hz are needlessly long; fixed windows ease sync and training.", { bullet: true, breakLine: true }),
          T("Interactions are sparse, so non-social windows still carry information.", { bullet: true, breakLine: true }),
          T("Event-centered windows show transitions before, during and after contact.", { bullet: true })],
    0.5, 5.4, 6.6, 1.8, { fontSize: 13 });
  // open questions
  rbox(s, 7.45, 3.95, 5.4, 2.65, OPEN_T, OPEN, { dash: true, lw: 1.25 });
  pill(s, 7.6, 4.07, "OPEN", WHITE, OPEN);
  txt(s, "Open questions before finalizing", 8.65, 4.05, 4.1, 0.35, { fontSize: 15, bold: true, color: OPEN });
  const qs = ["Physical states: hand-labeled, automatically discovered, or both?", "Best window length (testing ~10–30 s)?", "How to use the differing baseline structures?", "Balance or weight social vs non-social windows in training?"];
  txt(s, qs.map((q2, i) => T(q2, { bullet: true, breakLine: i < qs.length - 1, paraSpaceBefore: 6 })), 7.65, 4.5, 5.05, 2.55, { fontSize: 14 });
  s.addNotes("Starting idea: fixed windows in the 10–30 s range, likely starting around 20 s. Some windows are centered on social events (5 s before, the interaction, and post-event context), others are sampled from non-social periods; the same model sees both. The interaction itself varies in length but the input window stays fixed, and the true onset and offset are kept inside it. Many windows per mouse are summarized (state occupancy, transition patterns, latent summaries, respiration response) into one mouse/session-level prediction. Baseline handling is not finalized.");
}

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f));
