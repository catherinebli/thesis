# 6. Firmware setup and data formats

## Hardware assumed by the default config

| Item | Setting |
|---|---|
| MCU | Teensy 4.1, microSD in the built-in slot (SDIO) |
| IMUs | 5 × Adafruit ICM-20948 (address 0x69 by default, 0x68 with the jumper) |
| Bus 0, `Wire` (SDA 18 / SCL 19) | PCA9548 at 0x70: ch0 forearm, ch1 upper arm, ch2 chest |
| Bus 1, `Wire1` (SDA 17 / SCL 16) | PCA9548 at 0x70: ch0 thigh, ch1 shank |
| Bus 2, `Wire2` (SDA 25 / SCL 24) | spare |

Change the wiring in `config/pipeline.yaml` under `imus:`. Set `mux: null` for an IMU wired straight to a bus. Bus, mux and address changes do **not** change the feature hash, so they don't invalidate trained weights.

## ICM-20948 configuration (`lib/hal/Icm20948Imu.cpp`)

Register addresses were checked against Adafruit_ICM20X and the SparkFun ICM-20948 library.

| Register | Bank | Value | Why |
|---|---|---|---|
| `WHO_AM_I` 0x00 | 0 | read 0xEA | probe |
| `PWR_MGMT_1` 0x06 | 0 | 0x80, then 0x01 | reset, then wake with automatic clock selection |
| `PWR_MGMT_2` 0x07 | 0 | 0x00 | all accel and gyro axes on |
| `GYRO_SMPLRT_DIV` 0x00 | 2 | 4 | 1125 / (1 + 4) = 225 Hz |
| `GYRO_CONFIG_1` 0x01 | 2 | `(dlpf<<3) \| (fs<<1) \| 1` = 0x27 | DLPF ~24 Hz, ±2000 dps |
| `ACCEL_SMPLRT_DIV_1/2` 0x10/0x11 | 2 | 0 / 4 | 225 Hz |
| `ACCEL_CONFIG` 0x14 | 2 | 0x25 | DLPF ~24 Hz, ±8 g |
| `ACCEL_XOUT_H` 0x2D | 0 | burst-read 12 bytes | accel xyz then gyro xyz, big-endian int16 |

The bank is selected through `REG_BANK_SEL` 0x7F, bits [5:4].

Scale factors live in `PipelineConfig.h`:

* $a\,[\mathrm{m/s^2}] = \text{counts} \times 9.80665 / 4096$
* $\omega\,[\mathrm{rad/s}] = \text{counts} \times (\pi/180) / 16.4$

The magnetometer and DMP are not used (raw polling, no heading).

## Build and flash

```bash
pip install -e python/                 # or just run the scripts; they find the package themselves
python scripts/gen_headers.py          # whenever config/pipeline.yaml changes
cd firmware && pio run -e teensy41 -t upload && pio device monitor
```

* **Serial output** (115200 baud, optional): one line per scored stroke, plus a status line every second with calibration state, mode, kcal and per-IMU failure counts. Press `q` to close the session cleanly.
* **LED:**
  * slow blink while calibrating (sit still for the first 3 s);
  * solid while running;
  * fast blink if an IMU or the SD card failed at start-up (estimation continues regardless).

**Laptop-only checks** need no Teensy toolchain:

```bash
make -C firmware/test/native replay hal-check
python -m pytest tests
```

## `subject.cfg` (SD card root, optional)

```
id=S07
mass_kg=64.5
height_m=1.75
age=22
sex=F
```

Missing keys fall back to `subject_defaults` in the config. The same format is parsed by `parseSubjectCfg()` (C++) and `parse_subject_cfg()` (Python).

## `raw_###.bin`

Little-endian and packed: a 32-byte header followed by 69-byte records, one per 10 ms tick.

| Offset | Header field | Type |
|---|---|---|
| 0 | magic `"REE1"` | char[4] |
| 4 | version (1) | uint16 |
| 6 | header_bytes (32) | uint16 |
| 8 | record_bytes (69) | uint16 |
| 10 | n_ch (30) | uint8 |
| 11 | n_imu (5) | uint8 |
| 12 | fs_hz (100) | uint16 |
| 14 | reserved | uint16 |
| 16 | feature_hash | uint32 |
| 20 | session number | uint32 |
| 24 | boot millis | uint32 |
| 28 | pad | uint8[4] |

| Offset | Record field | Type |
|---|---|---|
| 0 | `n`, the sample index (+1 per tick) | uint32 |
| 4 | `t_us`, `micros()` at read time (wraps every 71.6 min) | uint32 |
| 8 | raw counts: ax ay az gx gy gz × 5 IMUs, sensor frame | int16[30] |
| 68 | `err`: bit *i* = IMU *i* failed and was held; bit 7 = missed tick | uint8 |

* **Timing uses `n`, not `t_us`.** A slow loop iteration produces held `ERR_GAP` frames, so `n` stays locked to the 100 Hz timer.
* **Preallocated tail.** The file is preallocated to 64 MB. If power is cut without pressing `q`, the reader keeps records up to the first break in `n` and ignores the rest.
* **Data rate.** 6.9 KB/s, or about 25 MB per hour.

## `est_###.csv` (and the Python and replay output)

```
n0,n1,T_s,P_w,P_raw_w,valid,flag_high,kcal_total
1234,1534,3.0000,362.812345,362.812345,1,0,4.210000
...
# end,21000,16.490000
```

| Column | Meaning |
|---|---|
| `n0`, `n1` | Stroke spans samples `[n0, n1)` |
| `P_w` | Power billed for the stroke |
| `P_raw_w` | Model output before the floor and cap (`nan` if the stroke wasn't scored) |
| `valid = 0` | Too many held samples or not buffered; the last estimate was held |
| `flag_high = 1` | Capped at 2500 W |
