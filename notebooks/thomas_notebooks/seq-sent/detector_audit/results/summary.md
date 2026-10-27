## 1. Reproduction of your saved rank features
689/689 windows reproduced exactly (n_breaths and breathing_rate_hz identical).

## 2. Reference peaks (live signal) not found by BreathMetrics, by reference prominence setting
      n_ref_live  n_missed_live  n_bm_live  n_bm_only_live  missed_%  bm_only_%
frac                                                                           
0.1        26536           2076      24523              63       7.8        0.3
0.2        26352           1904      24523              75       7.2        0.3
0.3        26188           1752      24523              87       6.7        0.4
0.4        26002           1574      24523              95       6.1        0.4

Per recording at frac=0.2:
              Subject    RankLabel  n_ref_live  n_missed_live  missed_%
trial                                                                  
CM_1_1_d1_2       1_1  Subordinate        3727            325       8.7
CM_1_2_sub1_1     1_2     Dominant        2974            292       9.8
CM_2_3_d2_4       2_3  Subordinate        4395            374       8.5
CM_2_4_sub2_3     2_4     Dominant        3895            202       5.2
CM_3_5_d3_6       3_5  Subordinate        2858            160       5.6
CM_4_7_d4_8       4_7  Subordinate        4230            211       5.0
CM_4_8_sub4_7     4_8     Dominant        4273            340       8.0

## 3. Are the missed peaks real breaths?
Missed peaks below the session threshold (mean+std/2): 77%  (found peaks below it: 0%)
Prominence / noise floor: missed median 41x (IQR 29-73), found median 89x
Half-prominence width: missed median 48 ms, found median 52 ms

Gap test (24373 BreathMetrics intervals with clean neighbours):
  intervals with NO missed peak inside (k=0): n=23103, length / neighbouring IBI median 1.00 (IQR 0.90-1.13)
  intervals WITH k>=1 missed peaks: n=1270, raw ratio median 2.36; ratio/(k+1) median 1.05 (IQR 0.87-1.49); 56% within 0.75-1.33
  k=1 intervals: position of missed peak in interval median 0.50 (IQR 0.44-0.57); 0.5 = exact midpoint

Autocorrelation rate check (575 fully-live windows with a periodic ACF):
  your BreathMetrics rate: median error -11%, within 15% of ACF in 49% of windows
  corrected-detector rate: median error -9%, within 15% of ACF in 58% of windows

Semi-synthetic test (real 60 s stretch scaled down, whole session refit; truth = peaks at full scale):
        trial  scale  bm_recall_of_bm_truth  bm_recall_of_ref_truth  ref_recall_of_ref_truth  n_truth_bm  n_truth_ref
  CM_1_1_d1_2   1.00                  1.000                   0.800                    1.000         329          411
  CM_1_1_d1_2   0.50                  0.495                   0.397                    1.000         329          411
  CM_1_1_d1_2   0.35                  0.085                   0.071                    1.000         329          411
  CM_1_1_d1_2   0.25                  0.003                   0.002                    1.000         329          411
  CM_1_1_d1_2   0.15                  0.000                   0.000                    1.000         329          411
CM_4_8_sub4_7   1.00                  1.000                   0.822                    1.000         385          466
CM_4_8_sub4_7   0.50                  0.673                   0.556                    0.998         385          466
CM_4_8_sub4_7   0.35                  0.262                   0.217                    0.998         385          466
CM_4_8_sub4_7   0.25                  0.029                   0.024                    0.998         385          466
CM_4_8_sub4_7   0.15                  0.003                   0.002                    0.983         385          466

## 4. Per-window refit (BreathMetrics fit on each 3 s window alone)
Recall of reference peaks: session fit 93%, window refit 80%
Window refit peaks with no reference peak (extra): mean 0.06 per window

## 5. Impact on classifier features (same 689 windows)
                               spearman  median_change_pct  dom_minus_sub_d_orig  dom_minus_sub_d_corr
feature                                                                                               
n_breaths                          0.94               4.17                 -1.40                 -1.45
breathing_rate_hz                  0.94               2.25                 -1.43                 -1.45
mean_ibi_sec                       0.94              -2.21                  1.51                  1.61
cv_ibi                             0.82              -5.54                  0.74                  0.69
mean_inhale_dur_sec                0.83              -4.81                  0.36                  1.72
cv_inhale_dur                      0.34             -18.35                 -0.32                  2.51
mean_exhale_dur_sec                0.91              -1.16                  1.25                  1.25
cv_exhale_dur                      0.73             -12.59                 -0.37                 -0.18
ie_ratio                           0.58              -5.13                 -0.53                  1.94
mean_peak_insp_flow                0.90              -6.16                 -0.36                 -0.54
mean_peak_exp_flow                 0.87              -3.71                  0.18                 -0.05
cv_peak_insp_flow                  0.70             -35.66                  0.41                  0.40
mean_inhale_vol                    0.71             -12.83                  0.83                  1.71
mean_exhale_vol                    0.76              -0.60                  0.98                  1.33
mean_tidal_vol_proxy               0.88              -4.03                  0.92                  1.57
cv_inhale_vol                      0.58             -35.27                 -0.58                  0.98
ventilation_proxy                  0.97              -1.39                 -0.77                 -0.55
pct_breaths_with_inhale_pause      0.43            -100.00                  0.50                  0.40
mean_inhale_pause_dur_sec          0.42            -100.00                  0.28                 -0.36
cv_inhale_pause_dur               -1.00              75.79                 -2.06                 -0.53
inhale_pause_duty_cycle            0.43            -100.00                  0.26                 -0.02
pct_breaths_with_exhale_pause      0.83              -5.56                  1.28                  1.30
mean_exhale_pause_dur_sec          0.66             -26.32                  1.85                  0.38
cv_exhale_pause_dur                0.33               0.64                 -0.14                 -0.87
exhale_pause_duty_cycle            0.72             -26.17                  1.46                  0.48
inhale_duty_cycle                  0.55              -1.94                 -0.89                  1.04
exhale_duty_cycle                  0.60               3.08                 -1.55                 -2.17

Per mouse (mean over windows):
                     rate_orig  rate_corr  n_orig  n_corr  rate_underestimate_%  missed_%
Subject RankLabel                                                                        
1_1     Subordinate       6.35       6.82   18.73   20.36                  6.76      8.72
1_2     Dominant          5.41       5.65   15.39   16.62                  4.19      9.82
2_3     Subordinate       6.75       7.28   19.83   21.62                  7.33      8.51
2_4     Dominant          6.01       6.27   17.83   18.67                  4.15      5.19
3_5     Subordinate       7.67       8.08   22.74   24.11                  5.03      5.60
4_7     Subordinate       6.70       7.09   19.92   21.04                  5.54      4.99
4_8     Dominant          6.65       7.16   19.62   21.28                  7.10      7.96

## 6. Rank classifier, original vs corrected features (identical model and folds)
Leave-one-cage-out (PairID). Pooled out-of-fold balanced accuracy; chance = 0.50.

  Feature set: all 27 features
  original  LogReg       window bal_acc=0.434   mice correct by majority vote=3/7   permutation p=0.66 (35 mouse relabelings)
  original  RandomForest window bal_acc=0.504   mice correct by majority vote=4/7
  corrected LogReg       window bal_acc=0.642   mice correct by majority vote=5/7   permutation p=0.20 (35 mouse relabelings)
  corrected RandomForest window bal_acc=0.596   mice correct by majority vote=5/7

  Feature set: timing only (validated)
  original  LogReg       window bal_acc=0.563   mice correct by majority vote=5/7   permutation p=0.23 (35 mouse relabelings)
  original  RandomForest window bal_acc=0.548   mice correct by majority vote=4/7
  corrected LogReg       window bal_acc=0.538   mice correct by majority vote=5/7   permutation p=0.29 (35 mouse relabelings)
  corrected RandomForest window bal_acc=0.500   mice correct by majority vote=4/7