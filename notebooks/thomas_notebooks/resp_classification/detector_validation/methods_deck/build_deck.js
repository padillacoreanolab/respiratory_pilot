// Builds respiration_methods_deck.pptx from figures/ (run: node build_deck.js, with pptxgenjs on NODE_PATH)
const pptxgen = require("pptxgenjs");
const path = require("path");
const F = (n) => path.join(__dirname, "figures", n + ".png");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const INK = "262626", MUTE = "666666", BLUE = "1F5FA8";

function slide(title, fs = 34) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText(title, { x: 0.55, y: 0.3, w: 12.2, h: 0.65, fontSize: fs, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
  return s;
}
function rail(s, lines, x = 10.55, y = 1.4, w = 2.35, size = 15, gap = 0.48) {
  lines.forEach((t, i) => s.addText(t, { x, y: y + i * gap, w, h: 0.42, fontSize: size, color: INK, fontFace: "Cambria Math", margin: 0, isTextBox: true, valign: "middle" }));
}
function small(s, t, x, y, w, h = 0.3, size = 11, color = MUTE, opts = {}) {
  s.addText(t, Object.assign({ x, y, w, h, fontSize: size, color, margin: 0, isTextBox: true }, opts));
}
function img(s, n, x, y, w, ar) { s.addImage({ path: F(n), x, y, w, h: w * ar }); }

// 1
let s = slide("From Trodes recording to respiration H5");
const nodes = ["Trodes .rec", "trodesexport -analogio", "ECU_Ain1.dat + timestamps.dat", "H5 /resp"];
const nx = [0.55, 3.35, 6.55, 10.35], nw = [2.2, 2.7, 3.3, 2.4];
nodes.forEach((t, i) => {
  s.addText(t, { x: nx[i], y: 1.45, w: nw[i], h: 0.5, fontSize: 17, bold: true, color: INK, align: "center", margin: 0, isTextBox: true, valign: "middle" });
  if (i < 3) s.addShape(pres.shapes.LINE, { x: nx[i] + nw[i] + 0.08, y: 1.7, w: nx[i + 1] - nx[i] - nw[i] - 0.16, h: 0, line: { color: INK, width: 1.5, endArrowType: "triangle" } });
});
small(s, "separate export step (not run by the converter)", 3.1, 1.98, 3.2, 0.3, 11);
small(s, "20,000 Hz · signed int16 ADC counts", 6.55, 1.98, 3.3, 0.3, 13, BLUE, { align: "center" });
small(s, "Aim1: wireless *_merged.rec · BLA pilot: tethered plain .rec", 0.55, 2.3, 6, 0.3, 11);
img(s, "s1_raw_trace", 0.55, 2.55, 9.4, 4.6 / 9.6);
rail(s, ["Aim1 valence: /resp (N,1), float64", "Aim1 rank: /resp (N,), float32", "Cohort 2: /analog/…ECU_Ain1 + /timestamps"], 10.2, 2.75, 2.8, 12, 0.62);
img(s, "s1_duplicate_ticks", 10.25, 4.75, 2.6, 1.9 / 2.9);
small(s, "Aim1 merged: repeated-timestamp filler; index time ~+0.22%", 10.2, 6.5, 2.8, 0.5, 10);

// 2
s = slide("20 kHz raw signal → 400 Hz respiration");
img(s, "s2_preprocessing_stack", 0.55, 1.15, 9.75, 5.75 / 9.7);
rail(s, ["LP: Butterworth order 4", "f_c = 200 Hz; filtfilt", "D = 20,000 / 400 = 50", "resample_poly(1, 50)", "BP: 0.1–20 Hz; zero phase", "t[n] = n / 400"], 10.55, 2.2, 2.4, 15, 0.62);
[2.9, 4.75].forEach((y) => s.addShape(pres.shapes.LINE, { x: 10.4, y, w: 0, h: 0.9, line: { color: INK, width: 1.2, endArrowType: "triangle" } }));
small(s, "cached: workbench_cache/*.npz", 10.55, 6.75, 2.4, 0.3, 11);

// 3
s = slide("Which signal is excluded?  Frozen QC screen");
img(s, "s3_qc_main", 0.55, 1.1, 9.75, 2.75 / 9.7);
img(s, "s3_qc_zooms", 0.55, 3.95, 9.75, 2.55 / 9.7);
rail(s, ["1 s blocks · hop 0.25 s", "r_k = P95_k − P5_k", "m = median_k(r_k)", "rel_k = r_k / m", "full_k = (max_k − min_k) / m", "dead: rel < 0.10", "artifact: rel > 8 OR full > 5", "degraded: median₂₁(rel) < 0.50", "dead/artifact: ±10 s buffer"], 10.55, 1.25, 2.6, 13, 0.5);
small(s, "31.2 min excluded · 262.8 min retained · 22/30 recordings affected", 0.55, 6.65, 9, 0.35, 15, INK);
small(s, "FROZEN QC ≠ DETECTOR VALIDITY", 10.3, 6.8, 2.7, 0.3, 11, "B03A2E", { bold: true });

// 4
s = slide("Detector validity: flat dropout or quiet breathing?");
img(s, "s4_validity", 0.45, 1.1, 10.0, 5.7 / 9.9);
rail(s, ["1 s / 0.25 s", "rel > 8 → artifact (±0.25 s)", "rel < 0.03 → test periodicity", "max ρ(60–500 ms) < 0.5 → dropout", "otherwise keep + low_amp", "rel < 0.15 → low_amp", "valid run ≥ 1 s"], 10.6, 1.4, 2.5, 13, 0.62);
small(s, "validity segmentation → peak detection per valid segment", 10.6, 6.0, 2.5, 0.6, 11);

// 5
s = slide("Adaptive candidate peaks: prominence, not peak height", 30);
img(s, "s5_prominence", 0.55, 1.05, 9.0, 4.55 / 9.6);
img(s, "s5_peak_geometry", 0.55, 5.35, 3.2, 2.65 / 5.0);
rail(s, ["5-point centered mean · 400 Hz", "12.5 ms effective window", "LR(t) = P90 − P10 over 1 s", "LR every 0.1 s → interpolate", "need(t) = max(0.20 LR(t), 2N)", "prominence ≥ need", "width ≥ 15 ms (6 samples)", "spacing ≥ 60 ms (24 samples)", "prominence horizon: ±0.5 s"], 10.45, 1.3, 2.7, 13, 0.55);

// 6
s = slide("Two passes: provisional cycles → local baseline");
img(s, "s6_two_passes", 0.55, 1.1, 9.75, 5.75 / 9.7);
rail(s, ["hp = HP_1Hz(seg)", "pk_A = select_peaks(hp)", "m_ab = [1/(b−a)] Σ_(i=a)^(b−1) seg[i]", "center = (a+b)/2", "b(t) = linear interpolate(m_ab)", "y(t) = seg(t) − b(t)", "pk_B = select_peaks(y; scale=hp)"], 10.5, 1.5, 2.6, 13, 0.7);

// 7
s = slide("Trough validation: which peaks survive?");
img(s, "s7_trough_main", 0.55, 1.1, 9.75, 3.0 / 9.7);
img(s, "s7_trough_pairs", 0.55, 4.2, 9.75, 2.6 / 9.7);
rail(s, ["t_i = argmin_(p_i < t < p_(i+1)) y[t]", "L = lower(p_i, p_(i+1))", "Δ_i = y[L] − y[t_i]", "keep if Δ_i ≥ need(L)", "otherwise remove L; repeat", "recompute baseline + troughs"], 10.5, 1.5, 2.6, 13, 0.7);
small(s, "peaks · troughs · baseline · valid mask · rejected reasons", 0.55, 6.85, 9, 0.3, 12, INK);

// 8
s = slide("OLD vs NEW: same signal, different peaks");
small(s, "OLD · BreathMetrics", 0.9, 1.05, 5, 0.35, 17, INK, { bold: true });
small(s, "NEW · adaptive detector", 7.0, 1.05, 5, 0.35, 17, INK, { bold: true });
img(s, "s8_old_vs_new", 0.55, 1.4, 12.2, 4.75 / 12.2);
small(s, "OLD: session-wide mean + SD/2 cutoff   vs   NEW: local prominence + noise floor   ·   11_3 OLD cutoff ≈714; local breath amplitudes ≈100–260; transient ≈35,600", 0.55, 6.25, 12.2, 0.3, 12, INK);
small(s, "BLA cohort 2 · 11_3, third window: OLD 0 → NEW 13", 0.55, 6.6, 5, 0.3, 14, INK);
small(s, "Aim1: 30 sessions | 262.8 min analyzed | 111,872 peaks | 7.10 Hz", 5.6, 6.6, 7.2, 0.3, 14, INK, { align: "right" });
small(s, "Algorithm outputs; no hand-labeled accuracy claim", 7.8, 6.98, 5, 0.3, 11, MUTE, { align: "right" });

pres.writeFile({ fileName: path.join(__dirname, "respiration_methods_deck.pptx") }).then((f) => console.log("wrote " + f));
