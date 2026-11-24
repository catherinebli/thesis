# 1. The method and its math

This pipeline follows the data-driven approach of Slade et al., *Sensing leg movement enhances wearable monitoring of energy expenditure*, Nature Communications 12, 4312 (2021), and their open-source code and data ([simtk.org/projects/energy-est](https://simtk.org/projects/energy-est)). Their final wearable used IMUs on the shank and thigh of one leg; this project adapts the method to erg rowing with five IMUs on one side of the body.

The whole method is: **cut the motion into cycles → turn each cycle into a fixed-length vector → linear regression against steady-state metabolic power → integrate power over time.**

---

## Step 1: Segment the motion into cycles

A cycle (a gait cycle for Slade, a stroke here) runs from one repeatable kinematic event to the next. Slade detects peaks of the low-passed shank gyro (height ≥ 70 deg/s, at least 0.6 s apart). Here the event is the **leg-drive peak** of the shank's sagittal-plane gyro, $\omega_z$, found online by `StrokeDetector` (see [04_pseudocode.md](04_pseudocode.md)).

Cycle $n$ covers samples $[t_n, t_{n+1})$. One cycle produces one model input and one estimate.

Whatever event you choose, training and deployment must use **the same detector**. Slade's real-time script has a `shift_ind` "offset between IMU heelstrike detection and the previous GRF based method". That is exactly the bug you get when training data was segmented one way and the device segments another way.

## Step 2: Low-pass filter

A Butterworth low-pass of order $N = 4$ at $f_c = 6$ Hz, with $f_s = 100$ Hz:

```math
|H(f)|^2 = \frac{1}{1 + (f/f_c)^{2N}}
```

* **Slade** uses `filtfilt`, which runs the filter forward and then backward. The phase cancels (zero lag), but the magnitude is applied twice: the effective response is $|H(f)|^2$, which is −6 dB at 6 Hz. It is also non-causal, because each output needs samples from after it.
* **This repo** uses the same Butterworth **causally** (one forward pass, −3 dB at 6 Hz) on both the Teensy and the laptop. The outputs then match sample for sample. The cost is a constant group delay of about 70 ms at stroke frequencies (≈ 6.9 samples at 0.33 Hz). That is harmless as long as training uses the same filter.

Each second-order section is direct form II transposed, the same recurrence as `scipy.signal.sosfilt`:

```math
y = b_0 x + z_1,\qquad z_1 \leftarrow b_1 x - a_1 y + z_2,\qquad z_2 \leftarrow b_2 x - a_2 y
```

**Steady-state start.** The filter starts from steady state for the first sample, so gravity on the accelerometers does not ramp up from zero. For a constant input $u$, a section settles at $y = g u$ with $g = \frac{b_0+b_1+b_2}{1+a_1+a_2}$, and solving the recurrence for the states gives

```math
z_1 = (b_1 + b_2)\,u - (a_1 + a_2)\,y,\qquad z_2 = b_2\,u - a_2\,y .
```

## Step 3: Time-normalize each cycle to $B$ bins

Every cycle becomes 0–100 % of a cycle with $B = 30$ values per channel, whatever its duration. With $L$ samples in the stroke, bin $k$ averages

```math
\bar s_c[k] = \frac{1}{\lvert I_k\rvert}\sum_{n\in I_k} s_c[n],\qquad
I_k = \Big\{\Big\lfloor \tfrac{kL}{B}\Big\rfloor,\ \dots,\ \Big\lfloor \tfrac{(k+1)L}{B}\Big\rfloor - 1\Big\}
```

Slade uses FFT resampling (`scipy.signal.resample`) instead: FFT the cycle, keep the lowest frequencies, inverse FFT at length $B$. Bin means are simpler in C and every bin is non-empty because $L \ge 120 > B$. Both views lead to the same three consequences:

* **Periodicity.** FFT resampling treats the cycle as one period of a periodic signal, which a catch-to-catch stroke nearly is. If the start and end values of a segment differ, you get ringing at the edges.
* **Bins are harmonics.** $B$ bins keep about $B/2$ harmonics of the cycle, and harmonic $k$ sits at $k/T$ Hz.
  * Walking ($T \approx 1.1$ s): 15 harmonics reach about 13.6 Hz, well above the 6 Hz filter.
  * Rowing at 20 spm ($T = 3$ s): 15 harmonics reach 5 Hz.
  * Choosing $B$ is choosing how many stroke harmonics to keep. Pick it from your data with Parseval. If $X_k$ are the stroke's Fourier coefficients, the energy kept by $H$ harmonics is

    ```math
    E(H) = \frac{\sum_{\lvert k\rvert \le H} \lvert X_k\rvert^2}{\sum_k \lvert X_k\rvert^2}.
    ```

    If 99 % sits in the first 7 harmonics, 15 bins are enough and the feature count halves.
* **Duration is thrown away.** Every stroke is normalized to the same length, so its duration has to come back as a separate feature. Slade's paper includes stride duration with height and weight. Here the stroke period $T$ is feature 3.

## Step 4: Flatten into a feature vector

Constants go first, then the channels in a fixed order, each with its $B$ bins:

```math
\mathbf x = \big[\,1,\ m,\ h,\ T,\ \bar s_1[0..B-1],\ \dots,\ \bar s_{M}[0..B-1]\,\big],\qquad D = 4 + MB
```

Here $M = 30$ channels (5 IMUs × accel xyz + gyro xyz) and $B = 30$, so $D = 904$. Slade's deployed model was $1 + 2 + 12\times 30 = 363$.

## Step 5: Normalize (training only)

Slade z-scores each signal with one mean and one standard deviation, pooled over all bins and all training cycles:

```math
z_{n,j,b} = \frac{x_{n,j,b}-\mu_j}{\sigma_j},\qquad \mu_j = \frac{1}{NB}\sum_{n,b} x_{n,j,b}
```

* Pooling over bins, rather than normalizing each column separately, keeps the shape of each signal across the cycle.
* Constants are normalized over the set of *unique* values, so subjects with more cycles do not dominate.
* The statistics come from the training fold only.

The device never normalizes. The normalization is folded into the weights at export:

```math
\hat y = b + \sum_d w_d\,\frac{x_d - \mu_d}{\sigma_d}
       = \Big(b - \sum_d \frac{w_d\,\mu_d}{\sigma_d}\Big) + \sum_d \frac{w_d}{\sigma_d}\,x_d
```

This is `LinearModel.fold_normalization`, tested in `tests/test_session.py`.

## Step 6: The linear model

**OLS.** Minimize $\sum_n (y_n - w_0 - \mathbf w^\top \mathbf x_n)^2$. With $X_c$ the centered design matrix,

```math
\mathbf w = (X_c^\top X_c)^{-1} X_c^\top \mathbf y_c,\qquad w_0 = \bar y - \mathbf w^\top \bar{\mathbf x}.
```

**Ridge** (Slade's paper used ridge with regularization 1). Using the SVD $X_c = U\Sigma V^\top$,

```math
\mathbf w = (X_c^\top X_c + \alpha I)^{-1} X_c^\top \mathbf y_c
          = \sum_i \frac{\sigma_i}{\sigma_i^2 + \alpha}\,(\mathbf u_i^\top \mathbf y)\,\mathbf v_i .
```

Directions with $\sigma_i^2 \ll \alpha$ are shrunk to about zero. The effective degrees of freedom are $\sum_i \sigma_i^2/(\sigma_i^2+\alpha)$.

**Lasso.** Minimize $\frac{1}{2N}\lVert \mathbf y - X\mathbf w\rVert^2 + \alpha \lVert \mathbf w\rVert_1$. The weights come out sparse, so it doubles as sensor and bin selection.

> sklearn's Ridge loss is **not** divided by $N$ but Lasso's is, so their $\alpha$ values are not comparable.

**Labels.** Every cycle in a condition gets the same label: steady-state metabolic power from respirometry (Brockway equation, with gas flows in mL/s):

```math
P = 16.58\,\dot V_{O_2} + 4.51\,\dot V_{CO_2}\quad [\mathrm{W}]
```

The paper averages the last 2 min of 5-min conditions and the last 3 min of 6-min conditions, and trains on the last 50 strides of each.

## Step 7: Cross-validation (Slade's harness)

* Each fold holds out one subject **and** `cond_holdout` conditions.
* Training uses every other subject minus those conditions, so every test is a new person doing a condition nobody trained on.
* `x_dev = x_test`: there is no separate dev set. Pick hyperparameters with nested cross-validation, or the reported error is biased low.

## Step 8: Metrics

| Metric | Definition | Notes |
|---|---|---|
| Per-cycle MAPE | $\frac1N\sum_n \lvert\hat y_n - y_n\rvert / y_n$ | |
| Condition-averaged | Average the predictions within a condition, then relative error | Never worse than per-cycle within a condition (triangle inequality). Averaging $k$ strokes shrinks random error about $1/\sqrt k$, but not bias. |
| Shifted | Add $\delta_j = \bar y_j - \bar{\hat y}_j$ per subject | Uses the test subject's own labels, so it measures tracking after perfect calibration. **Not an accuracy number.** |
| Bias baseline | Predict each subject's true mean for every condition | If shifted ≈ bias, the model tracks nothing within a subject. |
| Ordering | For each of the $\binom n2$ condition pairs, label ≈ / < / > for truth and estimate (≈ if $\lvert a-b\rvert/\tfrac{a+b}{2} \le 4.2\%$) | Fraction of pairs that agree; Kendall-tau-like with ties. |

## Step 9: Deployment and energy

**Fold averaging.** Slade averages weights across folds. For a linear model that equals averaging the fold models' predictions, $\frac1F\sum_f \mathbf w_f^\top\mathbf x = \bar{\mathbf w}^\top\mathbf x$, so it is a cheap ensemble. It is exact only if every fold used the same normalization. This repo instead refits once on all subjects and folds the normalization in (Step 5).

**Estimate.** Each stroke's estimate is $\hat P = \max(\min(\mathbf w^\top\mathbf x,\ P_{\max}),\ P_{\text{rest}})$. The resting floor is Mifflin–St Jeor, converted from kcal/day to watts, times a resting factor:

```math
P_{\text{rest}} = (10m + 625h - 5a + s)\cdot\underbrace{\tfrac{4184}{86400}}_{0.048426}\cdot k_{\text{rest}}
```

Here $m$ is in kg, $h$ in m (so $625h = 6.25 \times$ cm), $s = +5$ for M and $-161$ for F, and $k_{\text{rest}} = 1.41$ is Slade's standing factor. Choose a seated value for the erg.

**Energy.** Strokes tile the time axis, and pauses longer than 8 s are billed at $P_\text{rest}$:

```math
E = \sum_n \hat P_n\,T_n + P_{\text{rest}}\,T_{\text{rest}},\qquad
\text{kcal} = \frac{E\,[\mathrm J]}{4184},\qquad
\text{kcal/h} = 0.8604\,\hat P\,[\mathrm W]
```
