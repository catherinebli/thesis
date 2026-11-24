# 3. Issues that will come up

## Protocol and labels (the biggest risk)

1. **One condition per person means no usable model.** If everyone rows at the same 20 spm and the same intensity, the number of labels $L$ equals the number of subjects. The model then only learns to predict from height and weight, and the ordering and shifted metrics mean nothing.
   * Use 3–4 submaximal effort levels at 20 spm with a fixed drag factor (the PM5 displays it), plus a seated rest baseline.
2. **Labels come from respirometry**, via $P = 16.58\,\dot V_{O_2} + 4.51\,\dot V_{CO_2}$ (watts, with gas flows in mL/s).
   * VO₂ takes a few minutes to settle. A typical design is about 6-minute bouts with the last ~3 minutes averaged.
   * Train only on strokes from that same window.
   * Stay submaximal: above threshold VO₂ never settles, and RER > 1 means anaerobic energy the equation cannot see.
3. **Observability.** IMUs measure motion, not force. If two conditions produce nearly the same kinematics (same rate, different damper), no model can separate them.
   * Vary effort at a fixed damper, so drive speed changes.
   * This is also where EMG earns its place later.
4. **Mass scaling.** Metabolic cost scales roughly *multiplicatively* with body mass, but the model treats weight as one additive term. It is worth testing a W/kg target multiplied back by mass.
   * A pool of lightweight rowers spans a narrow mass range, which leaves that term poorly identified.
5. **Define the target metric.**
   * Respirometry repeatability is a few percent, and that noise is shared by every stroke in a bout. It averages down with the number of bouts, not strokes.
   * State "<10 %" in Slade's metric: condition-averaged, leave-one-subject-out. Not the shifted error.

## Segmentation

6. **Mismatched stroke boundaries between training and deployment.** Manually entered PM5 stroke times give a constant phase offset (the PM5 senses the flywheel, the IMU sees a body event) plus human jitter. Every bin then means a different phase than the weights expect. Slade's `shift_ind` exists for exactly this.
   * Segment the training data with the same `StrokeDetector`.
   * Use PM5 timing only to validate stroke counts and the offset.
7. **Thresholds are set from pilot data.** `thr_min` in `config/pipeline.yaml` is a placeholder.
   * Record a pilot, run `scripts/plot_session.py`, and set it to about 40 % of a typical drive peak.
   * Flip `detector.sign` if the drive comes out negative.

## Features

8. **Units.** The firmware logs raw counts and converts them with datasheet sensitivities: 4096 LSB/g at ±8 g, 16.4 LSB/(deg/s) at ±2000 dps. Driver unit conventions can't creep in.
   * Slade's script assumed the driver returned deg/s: a 70 deg/s threshold, then × π/180. Mixing that up scales every gyro weight by 57.3.
9. **Mounting repeatability.** A linear model is not rotation-invariant. If a sensor is rotated by $R$, its readings become $R\mathbf x$ and the weights see mixed-up axes.
   * Fix orientation with the 3D-printed housings and use the same side for every participant.
   * Set `mount:` once per housing design.
10. **Gyro bias** drifts with temperature and adds a constant to every bin.
    * The firmware calibrates from a 3 s still window at power-on, retried if the person moves.
    * Over a periodic stroke the segment returns to the same orientation, so $\int_{\text{stroke}}\boldsymbol\omega\,dt \approx \mathbf 0$. The per-stroke gyro mean is therefore about the residual bias. Subtracting it is a cheap upgrade if drift shows up.
11. **Aliasing.** The ICM-20948 runs at 225 Hz with its internal low-pass at about 24 Hz (`dlpf_cfg: 4`). That is below the 50 Hz Nyquist limit of the 100 Hz sampling.
12. **Bin count.** Choose it with the Parseval energy check in [01](01_method_and_math.md). Changing `bins` changes the feature hash, so weights must be retrained.

## Model and validation (when training starts)

13. **No dev set** in Slade's harness. Choosing α, bins or the sensor subset by test error biases that error low. Use nested leave-one-subject-out cross-validation.
14. **Collinearity across bins.** Neighbouring bins of a 6 Hz low-passed signal are highly correlated, so $X^\top X$ is ill-conditioned. OLS weights come out jagged across bins; ridge smooths them.
15. **Normalization and deployed weights.** Refit once on all subjects and fold the normalization in with `LinearModel.fold_normalization`. Then run `scripts/gen_headers.py --weights ...`.
    * The firmware refuses to compile if the weights' feature hash differs from the config's.

## Hardware

16. **I2C bandwidth.** Each byte takes 9 clocks. One 12-byte accel+gyro burst with addressing is about 15 bytes, or about 135 clocks per IMU. Five IMUs come to about 690 clocks per sample:

    | Bus speed | Time per frame | Share of the 10 ms budget |
    |---|---|---|
    | 400 kHz | ≈ 1.7 ms | ~17 % |
    | 100 kHz | ≈ 6.9 ms | ~69 % |

    `i2c_hz: 400000` is the default.
17. **I2C over body-length cables** is the usual cause of a flaky bus. Fast mode needs a rise time $t_r \approx 0.847\,R_p C_b \le 300$ ns, where $R_p$ is the pull-up resistance and $C_b$ the bus capacitance.
    * Cable adds tens of pF per metre. At $C_b = 200$ pF, $R_p \le 1.8$ kΩ.
    * The floor is $R_p \ge (3.3 - 0.4)/3\,\text{mA} \approx 1$ kΩ.
    * The Adafruit breakout's 10 kΩ pull-ups add in parallel with any you fit (2.2 kΩ for ~1 m runs).
    * The PCA9548 isolates each channel's capacitance. The firmware also recovers a stuck bus (`I2cBus::recover`) and re-initialises an IMU after 100 consecutive failures.
18. **SD cards stall** for 100 ms or more on some writes. Frames go into a 32 KB RAM ring (about 4.7 s) and are written in 512-byte blocks only when the card is not busy.
    * A slow loop shows up as `ERR_GAP` frames with the same sample index spacing, never as stretched time.
19. **Clocks.** Crystal oscillators drift about 20–50 ppm, or roughly 0.1–0.2 s per hour. That is fine for minute-averaged labels. To align strokes with the PM5 or the metabolic cart, put a sharp tap at the start of each bout.

## Bugs in Slade's harness (why this repo reimplements instead of reusing it)

| Where | What happens |
|---|---|
| `LRModel.run` model constructors | `normalize=` was removed in sklearn 1.2, so the constructors raise a TypeError. |
| Paths | Windows-only `'\\json files\\'` style paths, plus `os.chdir` everywhere. |
| `norm_vals` parsing | With 2 entries, `norm_y` is never defined. With `norm_y` true, the code uses `y_mean`/`y_std`, which don't exist. |
| `computeErrors` | Setting plotting to true with an LR model raises a NameError (`X_train_cum`, `lm_list`, `features`). |
| `str2bool` | `v in ("true")` is a substring test, so `""` and `"t"` both count as true. |
| Bin count | 30 is hard-coded in several places, and the number of constants is inferred with `% 30`. |
| `reg_type` | Not a regularization setting: `"L2"` means normal mode, anything else is parsed as 3 integers. |
| `addNLterms` | Powers are applied after z-scoring, so fractional powers of negative values give NaN. |
| `real_time_est.py` | `online` is undefined, `stride_window` is never updated, and `cnt` overflows offline. |
| Data layout | Assumes sorted folders, the same number of conditions for every subject, and contiguous labels. |
