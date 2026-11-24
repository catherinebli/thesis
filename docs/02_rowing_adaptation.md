# 2. Adapting the method: 5 IMUs, one side, rowing

IMUs: **forearm, upper arm, chest, thigh, shank**, all on the same side (the chest is midline). Each is an Adafruit ICM-20948 read as accel xyz + gyro xyz, which gives $5 \times 6 = 30$ channels. With $B = 30$ bins and four constants ($1, m, h, T$), the feature vector has $D = 904$ entries.

## Compute is not the bottleneck

* **Inference** is one 904-term dot product per stroke (about every 3 s). That takes microseconds on the Teensy 4.1's 600 MHz Cortex-M7 with its double-precision FPU. The per-sample filtering is 30 channels × 2 biquads.
* **Training** ridge with $D = 904$ means solving a 904×904 system, which takes well under a second on a laptop.
* **The real cost of high dimension is statistical**, and it depends on the number of labels, not the number of strokes.

## Strokes are not labels

Write each stroke as its condition mean plus a deviation, $\mathbf x_{c,s} = \bar{\mathbf x}_c + \boldsymbol\varepsilon_{c,s}$, with the same label $y_{c,s} = y_c$ for every stroke in condition $c$. By definition of the mean, $\sum_s \boldsymbol\varepsilon_{c,s} = \mathbf 0$. Let $r_c = y_c - w_0 - \mathbf w^\top \bar{\mathbf x}_c$. Then

```math
\sum_c\sum_s \big(r_c - \mathbf w^\top \boldsymbol\varepsilon_{c,s}\big)^2
= \sum_c n_c\, r_c^2 \;-\; 2\sum_c r_c\,\mathbf w^\top \underbrace{\textstyle\sum_s \boldsymbol\varepsilon_{c,s}}_{=\,\mathbf 0} \;+\; \mathbf w^\top S_W\,\mathbf w
```

where $S_W = \sum_{c,s}\boldsymbol\varepsilon_{c,s}\boldsymbol\varepsilon_{c,s}^\top$ is the within-condition scatter. So the training loss is exactly

```math
\mathrm{Loss}(\mathbf w) = \underbrace{\sum_c n_c\,\big(y_c - w_0 - \mathbf w^\top \bar{\mathbf x}_c\big)^2}_{\text{fit to } L \text{ condition means}}
\;+\; \underbrace{\mathbf w^\top S_W\, \mathbf w}_{\text{built-in penalty}} .
```

The fit term has only $L$ = subjects × conditions equations. More strokes add no equations. They add a penalty against weights on directions that vary from stroke to stroke.

With 12 subjects × 4 intensities, $L = 48$ against $D = 904$. Ridge regression and feature reduction are required regardless (fewer bins, gyro only, sensor subsets).

## "Extrapolating to both sides" comes for free

On the erg both sides move in phase: the handle is rigid and there is one seat. Model the other side as a sign-flipped copy with no phase shift, $\mathbf x_R \approx M\mathbf x_L$, where $M$ is diagonal with entries $\pm 1$. A bilateral linear model then collapses to a one-sided one:

```math
\hat y = w_0 + \mathbf w_L^\top \mathbf x_L + \mathbf w_R^\top M \mathbf x_L
       = w_0 + \underbrace{(\mathbf w_L + M\mathbf w_R)}_{\tilde{\mathbf w}}{}^{\top} \mathbf x_L
```

* Regressing whole-body metabolic power on one side's features learns $\tilde{\mathbf w}$ directly. The factor of about 2 is absorbed into the weights.
* Appending mirrored copies as extra columns makes $[X_L,\ X_L M]$ rank-deficient. OLS then has infinitely many solutions with identical predictions.
* Ridge on the duplicated design is exactly one-sided ridge with $\alpha/2$. For a fixed sum $\mathbf u = \mathbf w_1 + M\mathbf w_2$, the penalty $\lVert\mathbf w_1\rVert^2 + \lVert\mathbf w_2\rVert^2$ is smallest at $\mathbf w_1 = \mathbf u/2$, $\mathbf w_2 = M\mathbf u/2$, where it equals $\lVert\mathbf u\rVert^2/2$.
* So mirroring doubles the dimension and adds no information. **The pipeline has no mirroring step.**

Compare Slade's `generateSymmetricData`. It builds the other leg by circularly shifting each cycle by half (bins 15–29, then 0–14), because walking legs are 50 % out of phase. On the erg the shift is 0 %, so the "other side" would be a pure copy.

### If you ever do synthesize the other side

Accel and gyro need **different** sign flips under a reflection $R$ with $\det R = -1$:

```math
\mathbf a' = R\,\mathbf a \quad\text{(true vector)},\qquad
\boldsymbol\omega' = (\det R)\,R\,\boldsymbol\omega = -R\,\boldsymbol\omega \quad\text{(pseudovector)}.
```

With $z$ mediolateral, $R = \mathrm{diag}(1, 1, -1)$:

* Accel flips only $a_z$.
* Gyro flips $\omega_x$ and $\omega_y$ and keeps $\omega_z$; sagittal flexion looks the same on both sides.
* Then compose with how the other sensor is physically mounted.

Slade's `[1, -1, -1]` sign vector in `processRawGait` is applied identically to accel and gyro, and $\det = +1$. It is a 180° rotation about $x$ (a mounting fix), not a mirror. `config/pipeline.yaml` rejects mount matrices with $\det = -1$ for this reason.

### Where the symmetry assumption breaks

Sweep rowing on water, sailing, and single-arm lifts. The erg, bench press and shoulder press are fine.

## Detector choice for rowing

The shank's sagittal angular velocity has one large positive pulse per stroke (the leg drive) and a smaller negative dip on the recovery. The detector:

* tracks the drive peak with an adaptive threshold, $\max(\texttt{thr\_min},\ 0.5 \times \text{median of the last 5 peaks})$, so it follows intensity changes;
* confirms a peak once the signal falls below half the threshold;
* ignores peaks closer than 1.2 s (50 spm);
* treats gaps longer than 5 s (12 spm) as the start of a new bout.

Slade's real-time buffer was 4 s, which fails below about 16 spm. Here the ring buffer holds 10 s.
