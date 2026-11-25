# thesis
Full Body Energy Expenditure Wearable

## Branch `Barry-pre-realtime`: offline pipeline for the 5-IMU suit

This is the proof-of-concept version for the design review. Each erg piece is recorded over USB and
then processed on a laptop. Estimating on the Teensy in real time is the end goal and lives on `Barry`.

It is Catherine's pipeline (`ml/imu_preprocessing.py`, `ml/stroke_detection.py`,
`ml/erg_stroke_sync.py`) extended from one IMU to the suit's five: forearm, upper arm, chest, thigh and shank,
on one side of the body.

```
imu_identify.ino      check which IMU is on which mux channel (once, after wiring changes)
icm20948_mux5.ino     Teensy firmware: reads all 5 IMUs at 100 Hz
record_piece.py       saves one piece over USB  ->  data/raw/S01/S01_P1_imu.csv
Concept2 Logbook      stroke export for the same piece  ->  data/raw/S01/S01_P1_pm5.csv
build_dataset.py      every piece -> strokes synced to the PM5, 30 bins x 30 channels
                      -> out/dataset.csv, out/pieces_qc.csv, out/figures/
train_model.py        ridge model IMU -> PM5 power, leave-one-subject-out
                      -> out/model/ (errors, figures, weights)
```

### Setup

```
pip install -r requirements.txt
```

Arduino IDE with Teensyduino. `icm20948_mux5` needs the Adafruit ICM20X, Adafruit BusIO and Adafruit
Unified Sensor libraries. `imu_identify` needs none.

### Wiring check

Upload `firmware/imu_identify` and open the Serial Monitor. It lists what is on every mux channel. Type
`w` for the guided check, which moves one body part at a time and prints the `IMUS[]` table to paste into
`firmware/icm20948_mux5/icm20948_mux5.ino` if the wiring doesn't match it.

Current wiring (guided check, 2026-11-25), already set in both sketches:

| IMU | mux | channel |
|---|---|---|
| thigh | A (`Wire`, SDA 18 / SCL 19) | 2 |
| shank | A | 3 |
| chest | B (`Wire1`, SDA 17 / SCL 16) | 5 |
| forearm | B | 6 |
| upper arm (`upperarm`) | B | 7 |

### Recording a piece

1. Upload `firmware/icm20948_mux5`. Plug the suit into the laptop and close the Arduino Serial
   Monitor and Plotter.
2. Set the drag factor on the erg and keep it the same for every rower.
3. Run, with this piece's subject, piece id, intensity and drag factor:
   `python tools/record_piece.py S01 P1 --intensity easy --drag 120 --seconds 250`
4. **Sit still at the catch until it says ROW** (10 s). This lead-in is where each IMU's gyro offset
   is measured, and the first pull after it is what lines the IMU clock up with the PM5.
5. On ROW, start rowing in **Just Row** so the PM5 timer starts with the first pull. Row at 20 spm at
   the piece's intensity until the recording ends. `--seconds` includes the lead-in; the maximum is 600.
6. Save the piece to the Concept2 Logbook (ErgData app or the PM5's USB stick), download its stroke
   data CSV, and save it as `data/raw/S01/S01_P1_pm5.csv`.
7. Rest, then record the next intensity as a new piece (P2, P3 ...) with its own still start.

Add every rower to `data/subjects.csv` (copy `data/subjects_template.csv`). The recorder fills in
`data/pieces.csv`. See `data/README.md` for the columns.

### Processing and training

```
python ml/build_dataset.py                 # or --subject S01 for just the pilot
python ml/train_model.py --ablation        # needs 3+ subjects; with fewer it holds out pieces
```

What to look at:

- **`out/pieces_qc.csv`**: one row per piece. Anything listed under `review` needs a look.
  - `start_count_ok = False`: the PM5 labels may be shifted by one stroke. Check the piece's sheet.
    If needed, set `offset_s` in `pieces.csv` (the IMU time at which the PM5 timer read 0).
  - `resid_std_ms` up to about 150 ms is normal, because the PM5 rounds stroke times to 0.5 s.
  - `no_lead_in_bias`: the rower didn't sit still first, so that IMU's gyro offset wasn't removed.
- **`out/figures/<subject>_<piece>.png`**: the sync at the start of the piece, PM5 power per stroke,
  and the average stroke of every IMU.
- **`out/model/`**:
  - `metrics.json`: error per stroke and per piece (mean power) for the IMU model and two baselines
    (training mean; weight + height only).
  - `pred_vs_pm5.png`, `pieces.png`: predicted vs PM5 power for held-out strokes and pieces.
  - `ablation.png`: which IMUs carry the signal.
  - `weights.csv`: the final model in Slade's layout `[bias, weight, height, features...]`, with the
    order in `features.txt`.

Every prediction in the reported errors comes from a model that never saw that subject.

### Trying it without the suit

```
python tools/make_synthetic_data.py --out data_synth
python ml/build_dataset.py --data data_synth --out out_synth
python ml/train_model.py --dataset out_synth/dataset.csv --out out_synth/model --ablation
```

The synthetic power formula is made up, so these numbers show that the code works, not how good the
suit is.

### Changes from the `catherine` branch

- **New:**
  - `ml/multi_imu.py`: splits the suit CSV per IMU, cleans each with `clean_imu`, finds strokes on the
    thigh, and bins all five IMUs with the same boundaries.
  - `ml/build_dataset.py`, `ml/train_model.py`, `ml/suit_plots.py`.
  - `tools/record_piece.py`, `tools/make_synthetic_data.py`.
  - `firmware/`, data templates, `requirements.txt`.
- **`ml/erg_stroke_sync.py`:**
  - `synced_features` takes `cols=` (the default still bins one IMU).
  - `sync_strokes` raises a clear error when no PM5 stroke matches, instead of a `KeyError`.

  The existing notebooks run unchanged.
- **Gyro offset:** `multi_imu.lead_in_bias` measures it from the still lead-in using how much the gyro
  varies, not its size. `find_still` needs |ω| < 0.1 rad/s, and the ICM-20948's offset alone can be
  several deg/s.

### Limits of the proof of concept

- **Target:** the model predicts PM5 power, which is mechanical. kcal = Σ P·T / 0.25, the same ×4 the PM5
  uses but without its +300 kcal/h resting term. Checking against metabolic cost needs indirect
  calorimetry.
- **Feature count:** about 900 features against roughly 10 rowers. Ridge regression plus leave-one-subject-out keeps the
  reported error honest, but it should improve with more rowers.
- **Mounting:** the features use raw sensor axes, so each housing has to sit the same way on everyone.
