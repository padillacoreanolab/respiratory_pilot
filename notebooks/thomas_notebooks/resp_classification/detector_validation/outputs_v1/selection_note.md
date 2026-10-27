# Parameter selection (development data = odd minutes of the 7 rank recordings)

Pre-specified rule (sweep_dev.py docstring): mean amp_recall >= 0.97, spike_fp <= 0.05,
drift_stable >= 0.97, then minimise doubling + halving. **No config met it**: spike_fp was
~0.21 for every config.

Decomposition (CM_4_8, CM_1_2; 100 spikes of full-breath amplitude each): most 'spike
detections' were real breaths whose peak moved onto a nearby spike (12/16 and 23/34); net
extra breaths were +4 and +11 per 100 spikes. The metric conflated displacement with false
breaths and did not discriminate between configs, so it was dropped from selection
AFTER seeing results (disclosed post-hoc change) and is reported as a limitation.

Revised rule: worst-recording amp_recall >= 0.97 and worst-recording drift_stable >= 0.95;
minimise doubling + halving. Only config 20 qualified -> selected (noise_mult 3 -> 2; all other
parameters = prototype values). Prominence fraction 0.10-0.30 changed scores by < 5%;
0.08 s minimum spacing produced IBI pile-up at the limit (1.6%), 0.04-0.06 s none;
removing the 1 Hz scale high-pass lowered worst-case drift stability to 0.88.
