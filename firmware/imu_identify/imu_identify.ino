/*
  imu_identify.ino  --  which body segment is on which mux channel?
  Teensy 4.1 + 2x Adafruit 8-channel I2C mux (0x70) + Adafruit ICM-20948 IMUs.
  Only needs Wire (no Adafruit libraries).

  Wiring (Perma-Proto):
    mux A on Wire : SDA 18, SCL 19
    mux B on Wire1: SDA 17, SCL 16
  "A2" below means mux A, channel 2 (the SD2/SC2 port on the mux board).

  On start-up it scans both muxes, resets every IMU it finds, then measures the
  gyro offsets for 2 s: keep the suit still while it says so.

  Serial Monitor: type a letter, then Enter.
    l  live view (default): move one body part, it prints the channel rotating most
    w  guided check: one body part at a time, then prints the IMUS[] table for
       icm20948_mux5.ino
    p  plotter view: one trace per channel (Tools > Serial Plotter)
    s  scan again
    h  help
*/

#include <Wire.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

// ----------------------------------------------------------------- settings
const uint32_t I2C_HZ      = 100000;   // slow and safe on the long suit wires
const uint8_t  MUX_ADDR    = 0x70;
const uint32_t SAMPLE_US   = 20000;    // 50 Hz
const float    LIVE_DPS    = 20.0f;    // live view: slower than this counts as still
const float    MOVE_DPS    = 25.0f;    // guided check: minimum mean rotation for an answer
const float    CLEAR_RATIO = 2.0f;     // guided check: winner must beat the runner-up by this
const uint32_t STILL_MS    = 2000;
const uint32_t MOVE_MS     = 4000;

// The IMUS[] table in icm20948_mux5.ino (bus 0 = mux A / Wire, bus 1 = mux B / Wire1).
// A finished guided check overwrites bus/ch with what it found.
struct Segment { const char *name; uint8_t bus; uint8_t ch; const char *how; };
Segment SEG[] = {
  {"forearm",  1, 6, "Bend and straighten your ELBOW, over and over. Keep your upper arm still."},
  {"upperarm", 1, 7, "Swing your whole ARM forward and back from the shoulder, elbow straight."},
  {"chest",    1, 5, "Lean your TORSO forward and back, a bit quicker than normal."},
  {"thigh",    0, 2, "Sitting on a chair: lift your KNEE up and down from the hip."},
  {"shank",    0, 3, "Sitting on a chair: kick your lower leg out and back at the KNEE. "
                     "Keep your thigh on the seat."},
};
const uint8_t N_SEG = sizeof(SEG) / sizeof(SEG[0]);
// Guided order: the end of each limb first. Lifting the knee also swings the shank, so the
// shank must be known before the thigh (same for forearm before upper arm). Chest goes last.
const uint8_t ORDER[N_SEG] = {0, 1, 4, 3, 2};

// ---------------------------------------------------------------- ICM-20948
const uint8_t REG_BANK_SEL     = 0x7F;   // bits [5:4] = bank
const uint8_t B0_WHO_AM_I      = 0x00;   // reads 0xEA on an ICM-20948
const uint8_t B0_PWR_MGMT_1    = 0x06;   // bit 7 reset, bit 6 sleep, [2:0] clock
const uint8_t B0_PWR_MGMT_2    = 0x07;   // 0 = accel and gyro on
const uint8_t B0_GYRO_XOUT_H   = 0x33;   // gyro x, y, z: 6 bytes, big-endian
const uint8_t B2_GYRO_CONFIG_1 = 0x01;   // [5:3] filter, [2:1] range, [0] filter on
const uint8_t ICM_WHO_AM_I     = 0xEA;
const uint8_t GYRO_CFG         = (3 << 3) | (2 << 1) | 1;   // ~51 Hz filter, +/-1000 dps
const float   LSB_PER_DPS      = 32.8f;                     // at +/-1000 dps

// -------------------------------------------------------------------- buses
TwoWire *const BUS[2] = {&Wire, &Wire1};
const char BUS_TAG[2] = {'A', 'B'};
const char *const BUS_DESC[2] = {"Wire, SDA 18 / SCL 19", "Wire1, SDA 17 / SCL 16"};
const char *const BUS_CODE[2] = {"&Wire, ", "&Wire1,"};

// -------------------------------------------------------------------- state
struct Imu {
  uint8_t  bus, ch, addr;
  bool     ok;          // reset and configured
  bool     lastOk;      // last read worked
  float    bias[3];     // deg/s, from the last still period
  float    level;       // smoothed rotation rate, deg/s
  uint32_t errs;
};
const uint8_t MAX_IMU = 16;
Imu     imu[MAX_IMU];
uint8_t nImu = 0;
char    mode = 'l';     // 'l' live, 'p' plotter

// ------------------------------------------------------------------ helpers
bool probe(uint8_t b, uint8_t addr) {
  BUS[b]->beginTransmission(addr);
  return BUS[b]->endTransmission() == 0;
}

bool muxSet(uint8_t b, uint8_t mask) {            // bit n set = channel n connected
  BUS[b]->beginTransmission(MUX_ADDR);
  BUS[b]->write(mask);
  return BUS[b]->endTransmission() == 0;
}

bool writeReg(uint8_t b, uint8_t addr, uint8_t reg, uint8_t val) {
  BUS[b]->beginTransmission(addr);
  BUS[b]->write(reg);
  BUS[b]->write(val);
  return BUS[b]->endTransmission() == 0;
}

int readReg(uint8_t b, uint8_t addr, uint8_t reg) {
  BUS[b]->beginTransmission(addr);
  BUS[b]->write(reg);
  if (BUS[b]->endTransmission(false) != 0) return -1;
  if (BUS[b]->requestFrom(addr, (uint8_t)1) != 1) return -1;
  return BUS[b]->read();
}

const char *nameAt(uint8_t b, uint8_t ch) {        // what the table calls this channel
  for (uint8_t s = 0; s < N_SEG; s++)
    if (SEG[s].bus == b && SEG[s].ch == ch) return SEG[s].name;
  return "?";
}

// Full reset (clears whatever an earlier sketch set), wake up, gyro to +/-1000 dps.
bool icmInit(Imu &d) {
  if (!muxSet(d.bus, 1 << d.ch)) return false;
  writeReg(d.bus, d.addr, REG_BANK_SEL, 0x00);
  writeReg(d.bus, d.addr, B0_PWR_MGMT_1, 0x80);
  delay(20);
  for (uint8_t k = 0; k < 20; k++) {               // wait for the reset bit to clear
    int v = readReg(d.bus, d.addr, B0_PWR_MGMT_1);
    if (v >= 0 && !(v & 0x80)) break;
    delay(10);
  }
  delay(50);
  bool ok = writeReg(d.bus, d.addr, REG_BANK_SEL, 0x00) &&
            writeReg(d.bus, d.addr, B0_PWR_MGMT_1, 0x01) &&      // wake up, best clock
            writeReg(d.bus, d.addr, B0_PWR_MGMT_2, 0x00) &&      // accel and gyro on
            writeReg(d.bus, d.addr, REG_BANK_SEL, 2 << 4) &&
            writeReg(d.bus, d.addr, B2_GYRO_CONFIG_1, GYRO_CFG) &&
            readReg(d.bus, d.addr, B2_GYRO_CONFIG_1) == GYRO_CFG;
  writeReg(d.bus, d.addr, REG_BANK_SEL, 0x00);                   // data lives in bank 0
  delay(40);                                                     // gyro start-up
  d.bias[0] = d.bias[1] = d.bias[2] = 0;
  d.level = 0;
  d.errs = 0;
  d.lastOk = ok;
  return ok;
}

bool readGyro(Imu &d, float w[3]) {
  TwoWire *b = BUS[d.bus];
  bool ok = muxSet(d.bus, 1 << d.ch);
  if (ok) {
    b->beginTransmission(d.addr);
    b->write(B0_GYRO_XOUT_H);
    ok = b->endTransmission(false) == 0 && b->requestFrom(d.addr, (uint8_t)6) == 6;
  }
  if (ok) {
    for (uint8_t k = 0; k < 3; k++) {
      uint8_t hi = b->read(), lo = b->read();
      w[k] = (int16_t)((hi << 8) | lo) / LSB_PER_DPS;
    }
  } else {
    d.errs++;
  }
  d.lastOk = ok;
  return ok;
}

float rotation(Imu &d) {                           // deg/s, offset removed; -1 = read failed
  float w[3];
  if (!d.ok || !readGyro(d, w)) return -1;
  float x = w[0] - d.bias[0], y = w[1] - d.bias[1], z = w[2] - d.bias[2];
  return sqrtf(x * x + y * y + z * z);
}

// --------------------------------------------------------- scan and set up
void scanAndInit() {
  nImu = 0;
  Serial.println("\n# Scanning both muxes (A2 = mux A, channel 2)");
  for (uint8_t b = 0; b < 2; b++) {
    if (!probe(b, MUX_ADDR)) {
      Serial.printf("mux %c (%s): NOT FOUND at 0x70 -> check 3.3V, GND, SDA/SCL swap, solder joints\n",
                    BUS_TAG[b], BUS_DESC[b]);
      continue;
    }
    Serial.printf("mux %c (%s): found at 0x70\n", BUS_TAG[b], BUS_DESC[b]);
    for (uint8_t ch = 0; ch < 8; ch++) {
      const char *expect = nameAt(b, ch);
      bool expected = strcmp(expect, "?") != 0;
      if (!muxSet(b, 1 << ch)) {
        Serial.printf("  %c%u: mux did not accept the channel select\n", BUS_TAG[b], ch);
        continue;
      }
      uint8_t icms = 0, others = 0;
      bool printed = false;
      for (uint8_t a = 0x08; a <= 0x77; a++) {
        if (a == MUX_ADDR || !probe(b, a)) continue;
        bool isIcm = false;
        if (a == 0x68 || a == 0x69) {                // an upload mid-setup can leave it in bank 2
          writeReg(b, a, REG_BANK_SEL, 0x00);
          isIcm = readReg(b, a, B0_WHO_AM_I) == ICM_WHO_AM_I;
        }
        if (!printed) { Serial.printf("  %c%u:", BUS_TAG[b], ch); printed = true; }
        Serial.printf(" 0x%02X %s", a, isIcm ? "ICM-20948" : "(unknown device)");
        if (isIcm) {
          icms++;
          if (nImu < MAX_IMU) imu[nImu++] = Imu{b, ch, a, false, false, {0, 0, 0}, 0, 0};
        } else {
          others++;
        }
      }
      if (!printed && !expected) continue;                       // empty and not in the table
      if (!printed) Serial.printf("  %c%u: nothing", BUS_TAG[b], ch);
      if (expected) Serial.printf("   [mux5 table: %s]%s", expect, icms ? "" : "  <-- no IMU here");
      Serial.println();
      if (icms > 1) Serial.println("       ^ two IMUs on one channel: icm20948_mux5 expects one per channel");
    }
    muxSet(b, 0);                                                // all channels off
  }
  Serial.printf("# %u IMU(s) found, %u expected\n", nImu, N_SEG);
  for (uint8_t i = 0; i < nImu; i++) {
    imu[i].ok = icmInit(imu[i]);
    if (!imu[i].ok)
      Serial.printf("  %c%u: answered but would not configure, so it is skipped\n",
                    BUS_TAG[imu[i].bus], imu[i].ch);
  }
}

// -------------------------------------------------------------- measuring
void measureBias(uint32_t ms) {                    // average gyro while still = offset
  double   sum[MAX_IMU][3] = {};
  uint32_t cnt[MAX_IMU] = {};
  uint32_t t0 = millis(), next = micros();
  while (millis() - t0 < ms) {
    if ((int32_t)(micros() - next) < 0) continue;
    next += SAMPLE_US;
    for (uint8_t i = 0; i < nImu; i++) {
      float w[3];
      if (imu[i].ok && readGyro(imu[i], w)) {
        for (uint8_t k = 0; k < 3; k++) sum[i][k] += w[k];
        cnt[i]++;
      }
    }
  }
  for (uint8_t i = 0; i < nImu; i++)
    if (cnt[i]) for (uint8_t k = 0; k < 3; k++) imu[i].bias[k] = sum[i][k] / cnt[i];
}

void measureMotion(uint32_t ms, float mean[]) {    // mean rotation rate per IMU
  double   sum[MAX_IMU] = {};
  uint32_t cnt[MAX_IMU] = {};
  uint32_t t0 = millis(), dot = t0, next = micros();
  while (millis() - t0 < ms) {
    if (millis() - dot >= 500) { Serial.print('.'); dot += 500; }
    if ((int32_t)(micros() - next) < 0) continue;
    next += SAMPLE_US;
    for (uint8_t i = 0; i < nImu; i++) {
      float r = rotation(imu[i]);
      if (r >= 0) { sum[i] += r; cnt[i]++; }
    }
  }
  for (uint8_t i = 0; i < nImu; i++) mean[i] = cnt[i] ? sum[i] / cnt[i] : 0;
}

void calibrate() {
  Serial.println("# Keep the suit still for 2 s (measuring gyro offsets)...");
  measureBias(STILL_MS);
  for (uint8_t i = 0; i < nImu; i++) imu[i].level = 0;
  Serial.println("# done");
}

// ------------------------------------------------------------ live views
void printLive() {
  static uint32_t lastStill = 0;
  int top = -1;
  for (uint8_t i = 0; i < nImu; i++)
    if (imu[i].ok && (top < 0 || imu[i].level > imu[top].level)) top = i;
  if (top < 0) return;
  if (imu[top].level < LIVE_DPS) {
    if (millis() - lastStill >= 3000) { Serial.println("# still"); lastStill = millis(); }
    return;
  }
  lastStill = millis();
  Serial.printf(">> %c%u %-9s|", BUS_TAG[imu[top].bus], imu[top].ch,
                nameAt(imu[top].bus, imu[top].ch));
  for (uint8_t i = 0; i < nImu; i++) {
    if (!imu[i].ok) continue;
    if (!imu[i].lastOk) { Serial.printf("  %c%u  ERR         ", BUS_TAG[imu[i].bus], imu[i].ch); continue; }
    char bar[9];
    float q = imu[i].level / 40.0f;                // one # per 40 deg/s
    uint8_t len = q > 8 ? 8 : (uint8_t)q;
    memset(bar, '#', len);
    bar[len] = 0;
    Serial.printf("  %c%u %4.0f %-8s", BUS_TAG[imu[i].bus], imu[i].ch, imu[i].level, bar);
  }
  Serial.println();
}

void printPlot() {
  bool first = true;
  for (uint8_t i = 0; i < nImu; i++) {
    if (!imu[i].ok) continue;
    Serial.printf("%s%c%u_%s:%.1f", first ? "" : ",", BUS_TAG[imu[i].bus], imu[i].ch,
                  nameAt(imu[i].bus, imu[i].ch), imu[i].level);
    first = false;
  }
  if (!first) Serial.println();
}

// ----------------------------------------------------------- guided check
char waitKey(const char *allowed) {
  while (Serial.available()) Serial.read();        // drop anything typed earlier
  while (true) {
    if (!Serial.available()) continue;
    char c = tolower(Serial.read());
    if (c <= ' ') continue;
    delay(30);
    while (Serial.available()) Serial.read();      // rest of the line
    if (strchr(allowed, c)) return c;
    Serial.printf("(type one of: %s, then Enter)\n", allowed);
  }
}

bool picked(const int8_t *pick, uint8_t i) {
  for (uint8_t s = 0; s < N_SEG; s++) if (pick[s] == i) return true;
  return false;
}

void printResult(const int8_t *pick) {
  Serial.println("\n=== Result ===");
  bool all = true, same = true;
  for (uint8_t s = 0; s < N_SEG; s++) {
    if (pick[s] < 0) { Serial.printf("  %-9s not identified\n", SEG[s].name); all = false; continue; }
    const Imu &d = imu[pick[s]];
    Serial.printf("  %-9s = %c%u  (mux %c on %s, channel %u, address 0x%02X)\n", SEG[s].name,
                  BUS_TAG[d.bus], d.ch, BUS_TAG[d.bus], d.bus ? "Wire1" : "Wire", d.ch, d.addr);
    if (d.bus != SEG[s].bus || d.ch != SEG[s].ch) same = false;
  }
  if (!all) {
    Serial.println("\nNot every segment was identified. Fix that first (type w to run it again).");
    return;
  }
  if (same) {
    Serial.println("\nThis MATCHES the IMUS[] table in icm20948_mux5.ino. No change needed.");
    return;
  }
  Serial.println("\nThis does NOT match icm20948_mux5.ino. Replace its IMUS[] table with:\n");
  Serial.println("const ImuCfg IMUS[] = {");
  for (uint8_t s = 0; s < N_SEG; s++) {
    const Imu &d = imu[pick[s]];
    Serial.printf("  {\"%s\",%*s%s %u},\n", SEG[s].name, 9 - (int)strlen(SEG[s].name), "",
                  BUS_CODE[d.bus], d.ch);
  }
  Serial.println("};");
  for (uint8_t s = 0; s < N_SEG; s++) { SEG[s].bus = imu[pick[s]].bus; SEG[s].ch = imu[pick[s]].ch; }
  Serial.println("\n# The live view now uses these names.");
}

void wizard() {
  if (nImu == 0) { Serial.println("# No IMUs found. Fix the wiring, then type s to scan again."); return; }
  int8_t pick[N_SEG];
  for (uint8_t s = 0; s < N_SEG; s++) pick[s] = -1;
  Serial.println("\n=== Guided check ===");
  Serial.println("Wear the suit and sit on a chair within reach of the keyboard.");
  Serial.println("Each step: type g + Enter, hold still while it measures (2 s), then do the");
  Serial.println("movement from GO until STOP (4 s). Use the arm and leg that have sensors.");
  uint8_t step = 0;
  bool again = false;                              // repeat the step without asking first
  while (step < N_SEG) {
    uint8_t s = ORDER[step];
    if (!again) {
      Serial.printf("\n--- Step %u of %u: %s ---\n%s\n", step + 1, N_SEG, SEG[s].name, SEG[s].how);
      Serial.println(step ? "g = go, b = back one step, q = quit" : "g = go, q = quit");
      char c = waitKey(step ? "gbq" : "gq");
      if (c == 'q') { Serial.println("# guided check stopped"); return; }
      if (c == 'b') { step--; pick[ORDER[step]] = -1; continue; }
    }
    again = false;
    Serial.print("Hold still...");
    measureBias(STILL_MS);
    Serial.print(" GO! Keep moving until STOP");
    float mean[MAX_IMU];
    measureMotion(MOVE_MS, mean);
    Serial.println(" STOP");

    Serial.print("Mean rotation, deg/s:");
    for (uint8_t i = 0; i < nImu; i++)
      if (imu[i].ok) Serial.printf("  %c%u %.0f%s", BUS_TAG[imu[i].bus], imu[i].ch, mean[i],
                                   picked(pick, i) ? "*" : "");
    Serial.println("   (* = already identified)");

    int best = -1, second = -1;                    // among IMUs not identified yet
    for (uint8_t i = 0; i < nImu; i++) {
      if (!imu[i].ok || picked(pick, i)) continue;
      if (best < 0 || mean[i] > mean[best]) { second = best; best = i; }
      else if (second < 0 || mean[i] > mean[second]) second = i;
    }
    if (best < 0) {
      Serial.printf("No IMU left for %s, so it is skipped.\n", SEG[s].name);
      step++;
      continue;
    }
    bool moved = mean[best] >= MOVE_DPS;
    bool clear = moved && (second < 0 || mean[best] >= CLEAR_RATIO * mean[second]);
    if (clear) {
      pick[s] = best;
      Serial.printf("-> %s = %c%u (mux %c, channel %u, address 0x%02X)\n", SEG[s].name,
                    BUS_TAG[imu[best].bus], imu[best].ch, BUS_TAG[imu[best].bus], imu[best].ch,
                    imu[best].addr);
      step++;
      continue;
    }
    if (!moved)
      Serial.printf("Not enough movement: the most was %c%u at %.0f deg/s, it needs %.0f.\n",
                    BUS_TAG[imu[best].bus], imu[best].ch, mean[best], MOVE_DPS);
    else
      Serial.printf("Unclear: %c%u (%.0f) and %c%u (%.0f) moved about the same.\n",
                    BUS_TAG[imu[best].bus], imu[best].ch, mean[best],
                    BUS_TAG[imu[second].bus], imu[second].ch, mean[second]);
    Serial.printf("g = try again, a = accept %c%u anyway, q = quit\n", BUS_TAG[imu[best].bus], imu[best].ch);
    char c = waitKey("gaq");
    if (c == 'q') { Serial.println("# guided check stopped"); return; }
    if (c == 'a') {
      pick[s] = best;
      step++;
    } else {
      again = true;
    }
  }
  printResult(pick);
}

// ---------------------------------------------------------------- main
void printHelp() {
  Serial.println("\n# Commands (type the letter, then Enter):");
  Serial.println("#   l  live view: move one body part, it prints the channel rotating most");
  Serial.println("#   w  guided check: one body part at a time, then prints the IMUS[] table");
  Serial.println("#   p  plotter view (Tools > Serial Plotter), l to come back");
  Serial.println("#   s  scan again    h  this help");
  Serial.println("# Names next to a channel are what icm20948_mux5.ino currently calls it.");
  Serial.println("# Moving a whole limb turns both of its IMUs. To single one out, bend only the");
  Serial.println("# elbow (forearm) or kick only the knee while seated (shank).");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {}
  Serial.println("\n# imu_identify: which IMU is on which mux channel");
  Wire.begin();
  Wire.setClock(I2C_HZ);
  Wire1.begin();
  Wire1.setClock(I2C_HZ);
  scanAndInit();
  calibrate();
  printHelp();
}

void handleKeys() {
  while (Serial.available()) {
    char c = tolower(Serial.read());
    if (c == 'l') { mode = 'l'; Serial.println("# live view"); }
    else if (c == 'p') { mode = 'p'; }
    else if (c == 'w') { wizard(); mode = 'l'; }
    else if (c == 's') { scanAndInit(); calibrate(); }
    else if (c == 'h') { printHelp(); }
  }
}

void loop() {
  handleKeys();
  static uint32_t next = 0, lastPrint = 0;
  uint32_t now = micros();
  if ((int32_t)(now - next) < 0) return;
  next = ((int32_t)(now - next) > (int32_t)SAMPLE_US) ? now + SAMPLE_US : next + SAMPLE_US;
  for (uint8_t i = 0; i < nImu; i++) {
    float r = rotation(imu[i]);
    if (r >= 0) imu[i].level += 0.25f * (r - imu[i].level);    // ~80 ms smoothing
  }
  if (mode == 'p') {
    printPlot();
  } else if (millis() - lastPrint >= 200) {
    lastPrint = millis();
    printLive();
  }
}
