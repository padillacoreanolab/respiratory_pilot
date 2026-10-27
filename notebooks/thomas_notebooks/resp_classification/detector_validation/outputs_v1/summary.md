# Legacy vs adaptive_prominence on real recordings

Rank: evaluation minutes only (even minutes; odd minutes were used for parameter selection).
Valence: all minutes of every usable recording (never used for tuning). Excluded: 5_13 (documented drop; BLRI_5_13 has no resp dataset).

## Breath detection (valid evaluation samples)

**rank** (7 recordings, 34 valid minutes)
- breaths: legacy 12,788, adaptive 13,686 (+7.0%)
- found only by legacy: 3; only by adaptive: 901
- interval ~2x neighbours (missed-breath signature): legacy 4.0%, adaptive 3.3%
- interval <0.6x neighbours (split/false-peak signature): legacy 2.7%, adaptive 1.7%
- NeuroKit2 (khodadad2018) peaks also found: by legacy 95.8%, by adaptive 99.9%
- detector peaks also found by NeuroKit2: legacy 99.9%, adaptive 97.4%
- samples marked invalid: 2.68% (time-weighted)

**valence** (23 recordings, 224 valid minutes)
- breaths: legacy 83,849, adaptive 94,964 (+13.3%)
- found only by legacy: 21; only by adaptive: 11,136
- interval ~2x neighbours (missed-breath signature): legacy 4.4%, adaptive 3.3%
- interval <0.6x neighbours (split/false-peak signature): legacy 3.1%, adaptive 1.7%
- NeuroKit2 (khodadad2018) peaks also found: by legacy 91.8%, by adaptive 99.8%
- detector peaks also found by NeuroKit2: legacy 99.9%, adaptive 95.9%
- samples marked invalid: 1.17% (time-weighted)

### Per recording (rate = breaths per valid evaluation second)

| key           | cohort   |   subject | label       |   legacy_rate_hz |   adaptive_rate_hz |   legacy_only |   adaptive_only |   invalid_frac |   rate_change_pct |
|:--------------|:---------|----------:|:------------|-----------------:|-------------------:|--------------:|----------------:|---------------:|------------------:|
| CM_1_1_d1_2   | rank     |       1_1 | Subordinate |            6.147 |              6.704 |             0 |             164 |          0.104 |             9.056 |
| CM_1_2_sub1_1 | rank     |       1_2 | Dominant    |            4.785 |              5.358 |             0 |             171 |          0.067 |            11.975 |
| CM_2_3_d2_4   | rank     |       2_3 | Subordinate |            6.602 |              7.144 |             0 |             169 |          0.01  |             8.22  |
| CM_2_4_sub2_3 | rank     |       2_4 | Dominant    |            6.057 |              6.302 |             1 |              80 |          0     |             4.051 |
| CM_3_5_d3_6   | rank     |       3_5 | Subordinate |            8.039 |              8.339 |             0 |              54 |          0     |             3.732 |
| CM_4_7_d4_8   | rank     |       4_7 | Subordinate |            6.821 |              7.17  |             0 |             106 |          0     |             5.126 |
| CM_4_8_sub4_7 | rank     |       4_8 | Dominant    |            6.647 |              7.155 |             2 |             157 |          0     |             7.643 |
| BLRI_1_1      | valence  |       1_1 | BLRI        |            6.516 |              7.441 |             0 |             585 |          0     |            14.206 |
| BLRI_1_2      | valence  |       1_2 | BLRI        |            7.496 |              8.293 |             1 |             488 |          0     |            10.633 |
| BLRI_2_3      | valence  |       2_3 | BLRI        |            5.94  |              6.604 |             2 |             400 |          0.014 |            11.177 |
| BLRI_2_4      | valence  |       2_4 | BLRI        |            7.267 |              8.111 |             0 |             504 |          0.012 |            11.624 |
| BLRI_3_6      | valence  |       3_6 | BLRI        |            6.724 |              7.387 |             0 |             396 |          0.015 |             9.861 |
| BLRI_4_7      | valence  |       4_7 | BLRI        |            5.839 |              6.809 |             1 |             587 |          0     |            16.615 |
| BLRI_4_8      | valence  |       4_8 | BLRI        |            7.192 |              7.735 |             0 |             329 |          0     |             7.561 |
| RI1_1_1       | valence  |       1_1 | RI1         |            6.374 |              7.551 |             3 |             734 |          0     |            18.46  |
| RI1_1_2       | valence  |       1_2 | RI1         |            6.536 |              7.693 |             3 |             699 |          0.005 |            17.705 |
| RI1_2_3       | valence  |       2_3 | RI1         |            6.31  |              7.315 |             2 |             584 |          0.04  |            15.932 |
| RI1_2_4       | valence  |       2_4 | RI1         |            7.846 |              8.606 |             0 |             463 |          0     |             9.688 |
| RI1_3_5       | valence  |       3_5 | RI1         |            6.041 |              6.787 |             0 |             465 |          0     |            12.341 |
| RI1_3_6       | valence  |       3_6 | RI1         |            6.522 |              7.182 |             2 |             397 |          0.008 |            10.123 |
| RI1_4_7       | valence  |       4_7 | RI1         |            6.548 |              7.747 |             0 |             754 |          0     |            18.301 |
| RI1_4_8       | valence  |       4_8 | RI1         |            6.87  |              7.427 |             0 |             334 |          0.01  |             8.123 |
| RI2_1_1       | valence  |       1_1 | RI2         |            5.729 |              6.042 |             2 |             191 |          0     |             5.455 |
| RI2_1_2       | valence  |       1_2 | RI2         |            5.678 |              6.181 |             0 |             294 |          0.032 |             8.85  |
| RI2_2_3       | valence  |       2_3 | RI2         |            5.924 |              6.716 |             1 |             455 |          0.06  |            13.377 |
| RI2_2_4       | valence  |       2_4 | RI2         |            4.476 |              6.061 |             0 |             890 |          0.073 |            35.416 |
| RI2_3_5       | valence  |       3_5 | RI2         |            5.447 |              5.878 |             2 |              94 |          0     |             7.917 |
| RI2_3_6       | valence  |       3_6 | RI2         |            4.11  |              5.718 |             2 |             980 |          0.003 |            39.12  |
| RI2_4_7       | valence  |       4_7 | RI2         |            5.928 |              6.452 |             0 |             318 |          0     |             8.846 |
| RI2_4_8       | valence  |       4_8 | RI2         |            5.278 |              5.601 |             0 |             195 |          0     |             6.124 |

### Valence by condition (mean over recordings)

| label   |   legacy_rate_hz |   adaptive_rate_hz |   rate_change_pct |
|:--------|-----------------:|-------------------:|------------------:|
| BLRI    |             6.71 |               7.48 |             11.67 |
| RI1     |             6.63 |               7.54 |             13.83 |
| RI2     |             5.32 |               6.08 |             15.64 |

## By local amplitude (local 1 s range at the peak / recording median)

|                          |   legacy |   adaptive |   legacy_only |   adaptive_only |
|:-------------------------|---------:|-----------:|--------------:|----------------:|
| ('rank', '0.25-0.5x')    |      156 |        322 |             0 |             166 |
| ('rank', '0.5-1x')       |     5248 |       5769 |             3 |             526 |
| ('rank', '<0.25x')       |        8 |         99 |             0 |              91 |
| ('rank', '>1x')          |     7376 |       7496 |             0 |             118 |
| ('valence', '0.25-0.5x') |      691 |       2500 |             0 |            1807 |
| ('valence', '0.5-1x')    |    35340 |      41734 |            11 |            6393 |
| ('valence', '<0.25x')    |       29 |        945 |             0 |             916 |
| ('valence', '>1x')       |    47789 |      49785 |            10 |            2020 |

## Independent rate check (autocorrelation; no peak picking)

- legacy: median error -12.7%, within 15% in 49% of 2160 windows
- adaptive: median error -7.6%, within 15% in 66% of 2195 windows
  (the ACF reports the dominant period, which favours the larger of alternating breaths, so it is a conservative check for a detector that adds small breaths)

## Downstream window features, same windows (rank: your saved 3 s windows in evaluation minutes; valence: 100 windows/session from sample_random_nonoverlapping_windows, seed 42)

| feature                       | status            |    n |   spearman |   median_change_pct |   median_abs_change_pct |   legacy_nan_pct |   adaptive_nan_pct |
|:------------------------------|:------------------|-----:|-----------:|--------------------:|------------------------:|-----------------:|-------------------:|
| n_breaths                     | supported         | 2573 |      0.909 |               5.556 |                   5.556 |            2.093 |              0.266 |
| breathing_rate_hz             | supported         | 2573 |      0.896 |               4.67  |                   5.15  |            2.093 |              0.266 |
| mean_ibi_sec                  | supported         | 2573 |      0.896 |              -4.462 |                   4.914 |            2.093 |              0.266 |
| cv_ibi                        | supported         | 2567 |      0.769 |              -7.518 |                  15.536 |            2.321 |              0.381 |
| mean_inhale_dur_sec           | provisional       | 2573 |      0.734 |              -5.123 |                   8.492 |            2.093 |              0.266 |
| cv_inhale_dur                 | provisional       | 2573 |      0.477 |             -29.882 |                  41.531 |            2.093 |              0.304 |
| mean_exhale_dur_sec           | provisional       | 2573 |      0.899 |              -2.618 |                   5.463 |            2.093 |              0.266 |
| cv_exhale_dur                 | provisional       | 2573 |      0.769 |             -12.042 |                  20.468 |            2.093 |              0.342 |
| ie_ratio                      | provisional       | 2573 |      0.465 |              -3.319 |                  11.912 |            2.093 |              0.342 |
| mean_peak_insp_flow           | provisional       | 2573 |      0.894 |              -9.604 |                   9.682 |            2.093 |              0.266 |
| mean_peak_exp_flow            | provisional       | 2573 |      0.916 |              -0.999 |                   5.815 |            2.093 |              0.266 |
| cv_peak_insp_flow             | provisional       | 2573 |      0.702 |             -11.609 |                  19.934 |            2.093 |              0.266 |
| mean_inhale_vol               | proxy_provisional | 2573 |      0.666 |             -15.324 |                  15.534 |            2.093 |              0.266 |
| mean_exhale_vol               | proxy_provisional | 2573 |      0.817 |              -2.754 |                   8.326 |            2.093 |              0.342 |
| mean_tidal_vol_proxy          | proxy_provisional | 2573 |      0.822 |              -8.519 |                   8.553 |            2.093 |              0.342 |
| cv_inhale_vol                 | proxy_provisional | 2573 |      0.727 |             -17.916 |                  21.231 |            2.093 |              0.342 |
| ventilation_proxy             | proxy_provisional | 2573 |      0.959 |              -3.174 |                   3.966 |            2.093 |              0.342 |
| pct_breaths_with_inhale_pause | provisional       | 2573 |      0.362 |            -100     |                 100     |            2.093 |              0.266 |
| mean_inhale_pause_dur_sec     | provisional       | 2573 |      0.336 |            -100     |                 100     |            2.093 |              0.266 |
| cv_inhale_pause_dur           | provisional       |   21 |     -0.165 |             -68.922 |                  75.287 |           93.493 |             97.983 |
| inhale_pause_duty_cycle       | provisional       | 2573 |      0.348 |            -100     |                 100     |            2.093 |              0.266 |
| pct_breaths_with_exhale_pause | provisional       | 2573 |      0.624 |             -42.105 |                  66.667 |            2.093 |              0.266 |
| mean_exhale_pause_dur_sec     | provisional       | 2573 |      0.548 |             -56.132 |                  75.701 |            2.093 |              0.266 |
| cv_exhale_pause_dur           | provisional       |  375 |      0.217 |              -7.004 |                  49.968 |           75.419 |             75.381 |
| exhale_pause_duty_cycle       | provisional       | 2573 |      0.598 |             -57.79  |                  85.367 |            2.093 |              0.266 |
| inhale_duty_cycle             | provisional       | 2573 |      0.495 |              -0.455 |                   6.133 |            2.093 |              0.266 |
| exhale_duty_cycle             | provisional       | 2573 |      0.434 |               2.947 |                   6.745 |            2.093 |              0.266 |

rank: windows passing the >=5 breath gate: legacy 357/357, adaptive 357/357; windows with any invalid sample 0

valence: windows passing the >=5 breath gate: legacy 2198/2271, adaptive 2259/2271; windows with any invalid sample 27

## Whole-session sanity check (absolute units, compare across sessions)

| key         |   median_cycle_amplitude |   median_local_range |   noise_ptp |   invalid_frac |   low_amplitude_breath_frac |   n_breaths |
|:------------|-------------------------:|---------------------:|------------:|---------------:|----------------------------:|------------:|
| RI2_5_13    |                     4.28 |                 5.74 |        0.09 |           0.01 |                        0    |        3858 |
| RI1_5_13    |                     4.86 |                 6.46 |        0.64 |           0.24 |                        0.31 |        3411 |
| RI2_3_6     |                  1292.16 |              1232.97 |       13.01 |           0    |                        0    |        3478 |
| RI2_2_4     |                  1773.09 |              1705.64 |       12.9  |           0.07 |                        0.05 |        3403 |
| CM_1_1_d1_2 |                  1880.28 |              1762.93 |        9.27 |           0.1  |                        0    |        3737 |

## Polarity evidence (weak)

- median positive-lobe duration 50 ms vs negative-lobe 58 ms (negative longer in 100% of recordings)
- pauses after the negative lobe in 4.8% of breaths vs after the positive lobe 0.5% (median over recordings)
- Both fit positive = inspiration (mouse expiration is usually longer and pauses are mostly end-expiratory). The sensor's physical polarity is NOT recorded in the H5 metadata.