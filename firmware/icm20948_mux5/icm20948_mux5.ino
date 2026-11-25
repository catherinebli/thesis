/*
  icm20948_mux5.ino  --  5 IMUs on 2 muxes: waveform viewer + timed recorder
  Teensy 4.1 + 5x Adafruit ICM-20948 + 2x Adafruit 8-channel I2C mux (0x70)

  Wiring (Perma-Proto):
    Mux A (lower body) on Wire : SDA = 18, SCL = 19, 3.3V, GND  -> A2 thigh, A3 shank
    Mux B (upper body) on Wire1: SDA = 17, SCL = 16, 3.3V, GND  -> B5 chest, B6 forearm, B7 upper arm
    Both muxes stay at the default 0x70 because they are on separate buses.
    Edit the IMUS[] table below if your channels differ.

  Keys (Serial Monitor / Plotter message box, then Enter):
    1-5 = choose which IMU the plotter shows
    a = accel x/y/z (g)          g = gyro x/y/z (deg/s)
    n = |accel| (g)  -> ~1.00 at rest in any orientation
    s = toggle fixed y-scale lines on/off
    p = pause / resume the plotter stream
    i = reprint startup report (includes a per-channel I2C scan)
    r<sec> = record <sec> seconds (default 20, max 600) of all 5 IMUs as CSV, e.g. r20
             output: "#REC_BEGIN,..." line, CSV header, one row per frame, "#REC_END,..." line
    x = abort a recording

  Magnetometer and temperature are not logged (not used by the model).
*/

#include <Wire.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>

// ----------------------------- IMU map ------------------------------
struct ImuCfg {
  const char *name;   // used as the CSV column prefix
  TwoWire    *bus;    // &Wire = mux A, &Wire1 = mux B
  uint8_t     ch;     // mux channel 0-7 (printed on the mux board)
};

// Channels from imu_identify's guided check (2026-11-25). The order sets the CSV
// column order, which the pipeline and trained weights depend on: keep it.
const ImuCfg IMUS[] = {
  {"forearm",  &Wire1, 6},
  {"upperarm", &Wire1, 7},
  {"chest",    &Wire1, 5},
  {"thigh",    &Wire,  2},
  {"shank",    &Wire,  3},
};
const uint8_t N_IMU = sizeof(IMUS) / sizeof(IMUS[0]);

// ----------------------------- settings -----------------------------
const uint8_t  MUX_ADDR    = 0x70;
const uint32_t SAMPLE_HZ   = 100;                 // matches thesis logging rate
const uint32_t PERIOD_US   = 1000000UL / SAMPLE_HZ;
const uint32_t I2C_CLOCK   = 400000;              // drop to 100000 if long runs give errors
const uint8_t  GYRO_DIV    = 10;                  // gyro ODR  = 1100 / (1+10) = 100.0 Hz
const uint16_t ACCEL_DIV   = 10;                  // accel ODR = 1125 / (1+10) = 102.3 Hz
const uint8_t  GYRO_DLPF   = 4;                   // GYRO_DLPFCFG 4  -> 23.9 Hz bandwidth
// accel DLPF set through the library below:      ICM20X_ACCEL_FREQ_23_9_HZ
const uint32_t REC_DEFAULT_S = 20;
const uint32_t REC_MAX_S     = 600;

const float SCALE_ACCEL_G  = 2.0;
const float SCALE_GYRO_DPS = 500.0;

// --------------------------- ICM registers --------------------------
const uint8_t REG_BANK_SEL     = 0x7F;            // bits [5:4] select bank 0..3
const uint8_t B2_GYRO_CONFIG_1 = 0x01;            // [5:3] DLPFCFG, [2:1] FS_SEL, [0] FCHOICE
const uint8_t B2_ACCEL_CONFIG  = 0x14;            // [5:3] DLPFCFG, [2:1] FS_SEL, [0] FCHOICE
const uint8_t B0_ACCEL_XOUT_H  = 0x2D;            // start of accel+gyro data block

// Ranges and matching sensitivities (datasheet / Adafruit library values)
const icm20948_accel_range_t ACCEL_RANGE = ICM20948_ACCEL_RANGE_8_G;
const icm20948_gyro_range_t  GYRO_RANGE  = ICM20948_GYRO_RANGE_1000_DPS;
const float ACC_LSB_PER_G   = 4096.0f;            // +/-8 g
const float GYR_LSB_PER_DPS = 32.8f;              // +/-1000 dps

// ------------------------------ state -------------------------------
Adafruit_ICM20948 icm[N_IMU];
uint8_t  imuAddr[N_IMU];                          // 0x69 or 0x68, found at startup
uint32_t imuErrs[N_IMU];                          // failed mux selects / reads
float    ax[N_IMU], ay[N_IMU], az[N_IMU];         // m/s^2
float    gx[N_IMU], gy[N_IMU], gz[N_IMU];         // rad/s

uint8_t  plotImu    = 0;
char     mode       = 'a';
bool     showScale  = true;
bool     streaming  = true;
uint32_t nextUs     = 0;
uint32_t frames     = 0;
uint32_t rateWinMs  = 0;
uint32_t rateWinN   = 0;
float    measuredHz = 0;
uint32_t maxReadUs  = 0;                          // worst time to read all 5 IMUs

bool     recording  = false;
uint32_t recTarget  = 0;
uint32_t recRows    = 0;
uint32_t recLate    = 0;
uint32_t recMaxRead = 0;
uint32_t recErrs0   = 0;
uint32_t recT0us    = 0;

// ------------------------------- mux --------------------------------
const char *busName(TwoWire *bus) { return bus == &Wire ? "A/Wire" : "B/Wire1"; }

bool muxSelect(TwoWire *bus, uint8_t ch) {        // connect one channel, isolate the rest
  bus->beginTransmission(MUX_ADDR);
  bus->write((uint8_t)(1 << ch));
  return bus->endTransmission() == 0;
}

bool selectImu(uint8_t i) { return muxSelect(IMUS[i].bus, IMUS[i].ch); }

// ------------------------- raw register helpers ---------------------
// These act on whichever IMU's channel is currently selected.
bool writeReg(uint8_t i, uint8_t reg, uint8_t val) {
  TwoWire *b = IMUS[i].bus;
  b->beginTransmission(imuAddr[i]);
  b->write(reg);
  b->write(val);
  return b->endTransmission() == 0;
}

int readReg(uint8_t i, uint8_t reg) {
  TwoWire *b = IMUS[i].bus;
  b->beginTransmission(imuAddr[i]);
  b->write(reg);
  if (b->endTransmission(false) != 0) return -1;
  if (b->requestFrom(imuAddr[i], (uint8_t)1) != 1) return -1;
  return b->read();
}

bool setBank(uint8_t i, uint8_t bank) {
  return writeReg(i, REG_BANK_SEL, (bank & 0x03) << 4);
}

// Adafruit_ICM20X v2.0.7 enableGyrolDLPF() writes ACCEL_CONFIG instead of
// GYRO_CONFIG_1, so the gyro filter is set directly here. Keeps the range bits.
bool setGyroDLPF(uint8_t i, uint8_t cfg) {
  if (!selectImu(i) || !setBank(i, 2)) return false;
  bool ok = false;
  int v = readReg(i, B2_GYRO_CONFIG_1);
  if (v >= 0) {
    uint8_t nv = (uint8_t)((v & ~0x39) | ((cfg & 0x07) << 3) | 0x01);  // FCHOICE=1 enables DLPF
    ok = writeReg(i, B2_GYRO_CONFIG_1, nv) && (readReg(i, B2_GYRO_CONFIG_1) == nv);
  }
  setBank(i, 0);  // library expects bank 0 for data reads
  return ok;
}

void readConfigRegs(uint8_t i, int &gcfg, int &acfg) {
  gcfg = acfg = -1;
  if (!selectImu(i)) return;
  setBank(i, 2);
  gcfg = readReg(i, B2_GYRO_CONFIG_1);
  acfg = readReg(i, B2_ACCEL_CONFIG);
  setBank(i, 0);
}

// ----------------------------- reporting ----------------------------
void scanBus(TwoWire *bus, const char *label) {
  Serial.printf("# I2C scan %s:", label);
  bus->beginTransmission(MUX_ADDR);
  if (bus->endTransmission() != 0) {
    Serial.println(" mux 0x70 NOT found -> check 3.3V/GND, SDA/SCL swap, solder joints");
    return;
  }
  Serial.println(" mux 0x70 found");
  for (uint8_t ch = 0; ch < 8; ch++) {
    if (!muxSelect(bus, ch)) continue;
    Serial.printf("#   ch%u:", ch);
    uint8_t found = 0;
    for (uint8_t a = 1; a < 127; a++) {
      if (a == MUX_ADDR) continue;
      bus->beginTransmission(a);
      if (bus->endTransmission() == 0) { Serial.printf(" 0x%02X", a); found++; }
    }
    Serial.println(found ? "" : " -");
  }
  bus->beginTransmission(MUX_ADDR);                // deselect all channels
  bus->write((uint8_t)0);
  bus->endTransmission();
}

void scanAll() {
  scanBus(&Wire,  "mux A (Wire, SDA 18 / SCL 19)");
  scanBus(&Wire1, "mux B (Wire1, SDA 17 / SCL 16)");
}

void printReport() {
  Serial.printf("# I2C %lu Hz, loop target %lu Hz, measured %.1f Hz, worst 5-IMU read %lu us, frames %lu\n",
                (unsigned long)I2C_CLOCK, (unsigned long)SAMPLE_HZ, measuredHz,
                (unsigned long)maxReadUs, (unsigned long)frames);
  Serial.printf("# gyro ODR %.1f Hz, accel ODR %.1f Hz, accel +/-%d g, gyro +/-%d dps\n",
                1100.0f / (1 + GYRO_DIV), 1125.0f / (1 + ACCEL_DIV),
                2 << ACCEL_RANGE, 250 << GYRO_RANGE);
  for (uint8_t i = 0; i < N_IMU; i++) {
    int gcfg, acfg;
    readConfigRegs(i, gcfg, acfg);
    Serial.printf("# IMU %u %-8s mux %s ch%u addr 0x%02X  GYRO_CONFIG_1 0x%02X  ACCEL_CONFIG 0x%02X  errors %lu\n",
                  i + 1, IMUS[i].name, busName(IMUS[i].bus), IMUS[i].ch, imuAddr[i],
                  gcfg & 0xFF, acfg & 0xFF, (unsigned long)imuErrs[i]);
  }
  scanAll();
  Serial.printf("# plotting IMU %u (%s)\n", plotImu + 1, IMUS[plotImu].name);
  Serial.println("# keys: 1-5 a g n s p i | r<sec> record | x abort");
}

void fail(const char *msg) {
  pinMode(LED_BUILTIN, OUTPUT);
  while (true) {                       // fast blink + repeat the diagnosis every 2 s
    Serial.printf("# ERROR: %s\n", msg);
    scanAll();
    for (int k = 0; k < 20; k++) {
      digitalToggle(LED_BUILTIN);
      delay(100);
    }
  }
}

// ----------------------------- sampling -----------------------------
// One 12-byte burst from ACCEL_XOUT_H (bank 0): accel x/y/z then gyro x/y/z, big-endian.
// Done directly because the library's getEvent() returns true even when the bus read fails,
// and it also pulls 11 unused temp/mag bytes.
bool readImu(uint8_t i) {
  TwoWire *b = IMUS[i].bus;
  b->beginTransmission(imuAddr[i]);
  b->write(B0_ACCEL_XOUT_H);
  if (b->endTransmission(false) != 0) return false;
  if (b->requestFrom(imuAddr[i], (uint8_t)12) != 12) return false;
  int16_t r[6];
  for (uint8_t k = 0; k < 6; k++) {
    uint8_t hi = b->read(), lo = b->read();
    r[k] = (int16_t)((hi << 8) | lo);
  }
  const float A = SENSORS_GRAVITY_EARTH / ACC_LSB_PER_G;   // counts -> m/s^2
  const float W = SENSORS_DPS_TO_RADS / GYR_LSB_PER_DPS;   // counts -> rad/s
  ax[i] = r[0] * A; ay[i] = r[1] * A; az[i] = r[2] * A;
  gx[i] = r[3] * W; gy[i] = r[4] * W; gz[i] = r[5] * W;
  return true;
}

// Reads all 5 IMUs, one mux channel at a time.
// A failed IMU logs NaN for that frame and the others keep going.
void readAll() {
  for (uint8_t i = 0; i < N_IMU; i++) {
    if (!(selectImu(i) && readImu(i))) {
      ax[i] = ay[i] = az[i] = gx[i] = gy[i] = gz[i] = NAN;
      imuErrs[i]++;
    }
  }
}

uint32_t totalErrs() {
  uint32_t s = 0;
  for (uint8_t i = 0; i < N_IMU; i++) s += imuErrs[i];
  return s;
}

// ----------------------------- recording ----------------------------
uint32_t readUInt(uint32_t timeoutMs) {
  uint32_t v = 0, t0 = millis();
  bool any = false;
  while (millis() - t0 < timeoutMs) {
    int c = Serial.peek();
    if (c < 0) continue;
    if (c >= '0' && c <= '9') {
      v = v * 10 + (Serial.read() - '0');
      any = true;
      if (v > 100000) break;
    } else {
      break;
    }
  }
  return any ? v : 0;
}

void startRecording(uint32_t seconds) {
  if (seconds == 0) seconds = REC_DEFAULT_S;
  if (seconds > REC_MAX_S) seconds = REC_MAX_S;
  Serial.println();
  Serial.printf("#REC_BEGIN,seconds=%lu,sample_hz=%lu,n_imu=%u,accel_range_g=%d,gyro_range_dps=%d,"
                "gyro_odr_hz=%.1f,accel_odr_hz=%.1f,i2c_hz=%lu",
                (unsigned long)seconds, (unsigned long)SAMPLE_HZ, N_IMU,
                2 << ACCEL_RANGE, 250 << GYRO_RANGE,
                1100.0f / (1 + GYRO_DIV), 1125.0f / (1 + ACCEL_DIV), (unsigned long)I2C_CLOCK);
  for (uint8_t i = 0; i < N_IMU; i++)
    Serial.printf(",%s=%s:ch%u:0x%02X", IMUS[i].name, busName(IMUS[i].bus), IMUS[i].ch, imuAddr[i]);
  Serial.println();

  Serial.print("sample,t_s");
  for (uint8_t i = 0; i < N_IMU; i++) {
    const char *n = IMUS[i].name;
    Serial.printf(",%s_ax_mps2,%s_ay_mps2,%s_az_mps2,%s_gx_rads,%s_gy_rads,%s_gz_rads",
                  n, n, n, n, n, n);
  }
  Serial.println(",read_us");

  recTarget  = seconds * SAMPLE_HZ;
  recRows    = 0;
  recLate    = 0;
  recMaxRead = 0;
  recErrs0   = totalErrs();
  recording  = true;
  nextUs     = micros();
}

void endRecording(bool aborted) {
  recording = false;
  Serial.printf("#REC_END,rows=%lu,late=%lu,max_read_us=%lu,i2c_errors=%lu,aborted=%d\n",
                (unsigned long)recRows, (unsigned long)recLate, (unsigned long)recMaxRead,
                (unsigned long)(totalErrs() - recErrs0), aborted ? 1 : 0);
}

// ------------------------------- setup -------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  Serial.println("# icm20948_mux5 starting");

  Wire.begin();
  Wire1.begin();

  for (uint8_t i = 0; i < N_IMU; i++) {
    char msg[96];
    if (!selectImu(i)) {
      snprintf(msg, sizeof msg, "mux %s not answering (needed for %s)", busName(IMUS[i].bus), IMUS[i].name);
      fail(msg);
    }
    // Adafruit breakout default is 0x69; 0x68 if the SDO/ADR jumper is changed
    if (icm[i].begin_I2C(0x69, IMUS[i].bus, i))      imuAddr[i] = 0x69;
    else if (icm[i].begin_I2C(0x68, IMUS[i].bus, i)) imuAddr[i] = 0x68;
    else {
      snprintf(msg, sizeof msg, "%s not found on mux %s ch%u", IMUS[i].name, busName(IMUS[i].bus), IMUS[i].ch);
      fail(msg);
    }
    icm[i].setAccelRange(ACCEL_RANGE);
    icm[i].setGyroRange(GYRO_RANGE);
    icm[i].setGyroRateDivisor(GYRO_DIV);
    icm[i].setAccelRateDivisor(ACCEL_DIV);
    icm[i].enableAccelDLPF(true, ICM20X_ACCEL_FREQ_23_9_HZ);
    if (!setGyroDLPF(i, GYRO_DLPF))
      Serial.printf("# WARNING: %s gyro DLPF write did not verify\n", IMUS[i].name);
  }

  // begin_I2C() re-runs bus->begin(), which resets the clock, so set it last
  Wire.setClock(I2C_CLOCK);
  Wire1.setClock(I2C_CLOCK);

  printReport();
  pinMode(LED_BUILTIN, OUTPUT);
  nextUs = micros();
  rateWinMs = millis();
}

// -------------------------------- loop -------------------------------
void handleKeys() {
  while (Serial.available()) {
    char c = Serial.read();
    if (recording) {
      if (c == 'x') endRecording(true);
      continue;
    }
    if (c == 'a' || c == 'g' || c == 'n') mode = c;
    else if (c >= '1' && c < '1' + N_IMU) {
      plotImu = c - '1';
      Serial.printf("# plotting IMU %u (%s)\n", plotImu + 1, IMUS[plotImu].name);
    }
    else if (c == 's') showScale = !showScale;
    else if (c == 'i') printReport();
    else if (c == 'r') startRecording(readUInt(50));
    else if (c == 'p') {
      streaming = !streaming;
      Serial.println(streaming ? "# resumed" : "# paused (send p to resume)");
    }
  }
}

void loop() {
  handleKeys();
  if (!streaming && !recording) {
    nextUs = micros();
    rateWinN = 0;
    rateWinMs = millis();
    return;
  }

  uint32_t now = micros();
  if ((int32_t)(now - nextUs) < 0) return;
  bool late = false;
  nextUs += PERIOD_US;
  if ((int32_t)(now - nextUs) > (int32_t)PERIOD_US) {
    nextUs = now + PERIOD_US;
    late = true;
  }

  uint32_t t0 = micros();
  readAll();
  uint32_t dt = micros() - t0;
  if (dt > maxReadUs) maxReadUs = dt;
  frames++;

  rateWinN++;
  if (millis() - rateWinMs >= 1000) {
    measuredHz = rateWinN * 1000.0f / (millis() - rateWinMs);
    rateWinN = 0;
    rateWinMs = millis();
    digitalToggle(LED_BUILTIN);
  }

  if (recording) {
    if (recRows == 0) recT0us = t0;
    if (late && recRows > 0) recLate++;
    if (dt > recMaxRead) recMaxRead = dt;
    Serial.printf("%lu,%.6f", (unsigned long)recRows, (t0 - recT0us) * 1e-6);
    for (uint8_t i = 0; i < N_IMU; i++)
      Serial.printf(",%.4f,%.4f,%.4f,%.5f,%.5f,%.5f", ax[i], ay[i], az[i], gx[i], gy[i], gz[i]);
    Serial.printf(",%lu\n", (unsigned long)dt);
    if (++recRows >= recTarget) endRecording(false);
    return;
  }

  const float G = SENSORS_GRAVITY_EARTH;
  const float R2D = SENSORS_RADS_TO_DPS;
  uint8_t k = plotImu;

  switch (mode) {
    case 'a':
      Serial.printf("ax_g:%.3f,ay_g:%.3f,az_g:%.3f", ax[k] / G, ay[k] / G, az[k] / G);
      if (showScale) Serial.printf(",lo:%.1f,hi:%.1f", -SCALE_ACCEL_G, SCALE_ACCEL_G);
      break;
    case 'g':
      Serial.printf("gx_dps:%.2f,gy_dps:%.2f,gz_dps:%.2f", gx[k] * R2D, gy[k] * R2D, gz[k] * R2D);
      if (showScale) Serial.printf(",lo:%.0f,hi:%.0f", -SCALE_GYRO_DPS, SCALE_GYRO_DPS);
      break;
    case 'n':
      Serial.printf("anorm_g:%.3f", sqrtf(ax[k] * ax[k] + ay[k] * ay[k] + az[k] * az[k]) / G);
      if (showScale) Serial.print(",lo:0.0,hi:2.0");
      break;
  }
  Serial.println();
}
