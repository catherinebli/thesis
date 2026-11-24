# 4. Pseudocode: raw IMU counts → watts → kcal

The real implementation is `firmware/lib/core/*.cpp` and its mirror `python/rowing_ee/`. This page is the readable version. Names match the code.

> **The one rule:** training must call the same calibration, filter, detector and featurizer as the device. Weights are only valid for the exact filter, detector, bins, channel order and units they were trained with. The feature hash enforces this at compile time.

## 0. Config (`config/pipeline.yaml` → `PipelineConfig.h`)

```
FS = 100 Hz
IMUS = [forearm, upper_arm, chest, thigh, shank]   # fixed order, one side (chest midline)
CH   = [ax, ay, az, gx, gy, gz] per IMU            # 30 channels
B = 30 bins; D = 4 + 30*B = 904
LPF = Butterworth(order 4, 6 Hz) as 2 biquads      # designed in scipy, written to the header
DETECT = shank gz, sign +1; thr = max(THR_MIN, 0.5 * median(last 5 peaks)); HYST = 0.5
MIN_STROKE = 1.2 s (50 spm), MAX_STROKE = 5 s (12 spm), REST_TIMEOUT = 8 s, BUFFER = 10 s
K_REST = 1.41, P_MAX = 2500 W
```

## 1. Acquisition (Teensy, `src/main.cpp`)

```
setup():
    Wire, Wire1, Wire2 @ 400 kHz
    for each IMU in config order: Icm20948Imu(bus, addr 0x69, mux 0x70, channel).begin()
        # WHO_AM_I = 0xEA, reset, wake, ODR 225 Hz, DLPF ~24 Hz, +-8 g, +-2000 dps
    subject = subject.cfg on SD card, else config defaults
    open raw_###.bin + est_###.csv (preallocated, 32 KB RAM ring)
    pipeline.begin(subject)                    # calibration starts with the first frame
    clock.every(10 ms): ticks += 1             # ISR does nothing else

loop():
    while next_n < ticks:
        if next_n + 1 < ticks:                 # we fell behind: hold this missed tick
            frame = last_frame; frame.err = ERR_GAP
        else:
            frame.raw, frame.err = sensors.readAll()   # failed IMU: retry once, else hold + set bit
        logger.logRaw(frame)                   # always log raw counts
        if pipeline.onSample(frame, rec): logger.logStroke(rec)
        next_n += 1
    logger.service()                           # 512-byte block to SD only if the card isn't busy
    on serial 'q': kcal = pipeline.endSession(next_n); logger.close()
```

## 2. Per-sample processing (`Pipeline::onSample`)

```
onSample(frame):
    if first frame: energy.begin(frame.n)
    if not calibration.done:
        calibration.addStill(frame.raw)        # 3 s window; retried if any gyro std > 0.05 rad/s
        return
    x = calibration.apply(frame.raw)           # counts -> m/s^2, rad/s; minus gyro bias; R_MOUNT
    if first processed frame: filter.reset(x)  # steady-state start, no gravity transient
    y = filter.step(x)                         # causal Butterworth, DF-II-T
    ring.push(frame.n, float32(y), frame.err)
    event, n0, n1 = detector.step(DETECT_SIGN * y[shank_gz], frame.n)
    if event == FIRST:  energy.startBout(n0)
    if event == STROKE: onStroke(n0, n1)
    energy.tick(frame.n)
```

## 3. Stroke detection (`StrokeDetector::step`)

```
step(s, n):
    thr = THR_MIN if no peaks yet else max(THR_MIN, THR_FRAC * median(last 5 peaks))
    if s > thr and s > cand_val:  cand_val, cand_n = s, n          # climbing a drive peak
    elif cand_n set and s < HYST * thr:                           # fell off it -> confirmed
        if no previous peak:                     return FIRST(cand_n)
        elif cand_n - last_peak >= MIN_STROKE:   return STROKE(last_peak, cand_n)
        else: ignore it (double bump inside one stroke)
        (on FIRST/STROKE: last_peak = cand_n; remember cand_val)
        reset candidate
```

## 4. One stroke → feature vector (`Pipeline::onStroke`, `Featurizer::extract`)

```
onStroke(n0, n1):
    L = n1 - n0
    if energy.mode == REST or L > MAX_STROKE:
        energy.startBout(n1); return             # first stroke after a pause: no estimate
    valid = ring.contains(n0, n1) and (held samples in [n0, n1)) / L <= 5 %
    if valid:
        x = [1, mass, height, L/FS]
        for c in 0..29, k in 0..B-1:
            x.append(mean(ring[c][n0 + floor(k*L/B) : n0 + floor((k+1)*L/B)]))
        P_raw = dot(W, x)                        # W[0] = intercept, normalization folded in
        P = max(min(P_raw, P_MAX), P_rest)       # flag if capped
    energy.addStroke(n0, n1, P, valid)           # invalid -> hold last estimate
```

## 5. Resting floor

```math
P_{\text{rest}} = (10m + 625h - 5a + s)\cdot \tfrac{4184}{86400}\cdot K_{\text{rest}},\qquad s = +5\ (\text{M}),\ -161\ (\text{F})
```

## 6. Energy accounting (`EnergyMeter`)

```
begin(n):        E = 0; n_acc = n; mode = REST; P_last = P_rest
startBout(n):    E += P_rest * (n - n_acc)/FS; n_acc = n; mode = ROWING
addStroke(n0, n1, P, valid):
                 if not valid: P = P_last
                 E += P * (n1 - n_acc)/FS; n_acc = n1; if valid: P_last = P
tick(n):         if ROWING and n - last_peak > REST_TIMEOUT: mode = REST   # pause billed at P_rest
liveKcal(n):     (E + (P_last if ROWING else P_rest) * (n - n_acc)/FS) / 4184
end(n):          same, then return total kcal
```

```math
E = \sum_n \hat P_n T_n + P_{\text{rest}} T_{\text{rest}},\qquad \text{kcal} = E/4184
```

## 7. Drivers

```
Offline (Python):  log = RawLog.read("raw_000.bin"); res = Pipeline.run_session(log, cfg, model, subject)
Real time:         loop() above; the serial status line shows pipeline.liveKcal(next_n) every second
Parity:            scripts/check_parity.py replays the same log through the compiled C++ core
```
