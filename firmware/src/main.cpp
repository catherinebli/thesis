// Teensy 4.1 entry point: 5x ICM-20948 -> 100 Hz frames -> SD log + on-board
// energy estimate. All signal processing lives in lib/core (shared with the
// laptop replay); this file only wires hardware to it.
//
// Serial (115200, optional): one line per stroke, a status line every second.
//   q = end the session cleanly (flush + close files), then idle
#include <Arduino.h>
#include <Wire.h>
#include <string.h>

#include "FrameRing.h"
#include "I2cBus.h"
#include "Icm20948Imu.h"
#include "LinearModel.h"
#include "ModelWeights.h"
#include "Pipeline.h"
#include "PipelineConfig.h"
#include "SampleClock.h"
#include "SdLogger.h"
#include "SensorArray.h"
#include "Subject.h"
#include "Types.h"

static_assert(MW_D == PC_D, "ModelWeights.h and PipelineConfig.h disagree on D: run scripts/gen_headers.py");
static_assert(MW_FEATURE_HASH == PC_FEATURE_HASH,
              "weights were trained with a different feature config: retrain or regenerate headers");

using namespace ree;

static I2cBus bus0(Wire, 18, 19);
static I2cBus bus1(Wire1, 17, 16);
static I2cBus bus2(Wire2, 25, 24);
static I2cBus* const BUSES[3] = {&bus0, &bus1, &bus2};

DMAMEM static float ringData[PC_RING_CAP * PC_N_CH];   // ~117 KB in RAM2
DMAMEM static uint8_t ringErr[PC_RING_CAP];
static FrameRing ring(ringData, ringErr, PC_RING_CAP);
static LinearModel model(MW_W, MW_IS_PLACEHOLDER != 0);
static Pipeline pipeline(ring, model);

static Icm20948Imu* imus[PC_N_IMU];
static SensorArray sensors;
static SampleClock sampleClock;
static SdLogger logger;

static bool logging = false;
static bool running = false;
static bool hardwareFault = false;
static uint32_t nextN = 0;
static int16_t lastRaw[PC_N_CH];
static uint32_t lastStatusMs = 0;

static void printStroke(const StrokeRecord& r) {
    Serial.printf("stroke %lu-%lu  T=%.2fs  P=%.1fW%s%s  kcal=%.2f\n",
                  static_cast<unsigned long>(r.n0), static_cast<unsigned long>(r.n1), r.T_s, r.P_w,
                  r.valid ? "" : " (held)", r.flagHigh ? " (capped)" : "", r.kcalTotal);
}

static void printStatus() {
    Serial.printf("t=%.1fs cal=%s mode=%s kcal=%.2f err=[", nextN / static_cast<double>(PC_FS_HZ),
                  pipeline.calibrated() ? "ok" : "sit still", pipeline.mode() == EnergyMeter::Rowing ? "rowing" : "rest",
                  pipeline.liveKcal(nextN));
    for (int i = 0; i < PC_N_IMU; ++i)
        Serial.printf(i ? ",%lu" : "%lu", static_cast<unsigned long>(sensors.failures(i)));
    Serial.printf("] sd=%s overruns=%lu\n", logging ? logger.rawName() : "off",
                  static_cast<unsigned long>(logger.overruns()));
}

static void endSession() {
    const double kcal = pipeline.endSession(nextN);
    sampleClock.end();
    if (logging) logger.close(nextN, kcal);
    logging = false;
    running = false;
    Serial.printf("session closed: %.2f kcal over %.1f s\n", kcal, nextN / static_cast<double>(PC_FS_HZ));
}

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    const uint32_t t0 = millis();
    while (!Serial && millis() - t0 < 1500) {}

    bus0.begin(PC_I2C_HZ);
    bus1.begin(PC_I2C_HZ);
    bus2.begin(PC_I2C_HZ);
    for (int i = 0; i < PC_N_IMU; ++i) {
        const PcImuSpec& s = PC_IMUS[i];
        imus[i] = new Icm20948Imu(*BUSES[s.bus], s.addr, s.mux, s.channel, s.name);
        sensors.add(imus[i]);
    }
    const uint8_t failed = sensors.beginAll();
    for (int i = 0; i < PC_N_IMU; ++i)
        Serial.printf("IMU %-9s bus %u mux 0x%02X ch %u: %s\n", PC_IMUS[i].name, PC_IMUS[i].bus, PC_IMUS[i].mux,
                      PC_IMUS[i].channel, (failed >> i) & 1 ? "NOT FOUND" : "ok");
    hardwareFault = failed != 0;

    Subject subj = defaultSubject();
    if (logger.begin()) {
        if (!logger.readSubject(subj)) Serial.println("no subject.cfg on the card: using config defaults");
        logging = logger.openSession(PC_FEATURE_HASH, millis());
        Serial.printf("logging to %s\n", logging ? logger.rawName() : "(failed to open)");
    } else {
        Serial.println("SD card not found: estimating without logging");
    }
    hardwareFault |= !logging;

    pipeline.begin(subj);
    Serial.printf("subject %s  %.1f kg  %.2f m  age %.0f  %c  ->  resting floor %.1f W\n", subj.id, subj.massKg,
                  subj.heightM, subj.age, subj.sex, pipeline.restWatts());
    if (model.isPlaceholder())
        Serial.println("WARNING: placeholder weights, every stroke is billed at the resting floor");
    Serial.printf("sit still for %.1f s while the gyros calibrate\n", PC_STILL_N / static_cast<double>(PC_FS_HZ));

    memset(lastRaw, 0, sizeof(lastRaw));
    sampleClock.begin(PC_FS_HZ);
    running = true;
}

void loop() {
    if (running) {
        const uint32_t ticks = sampleClock.ticks();
        while (nextN < ticks) {
            RawFrame f;
            f.n = nextN;
            f.t_us = micros();
            if (nextN + 1 < ticks) {               // behind schedule: hold this missed tick
                memcpy(f.raw, lastRaw, sizeof(lastRaw));
                f.err = ERR_GAP;
            } else {
                f.err = sensors.readAll(f.raw);
                memcpy(lastRaw, f.raw, sizeof(lastRaw));
            }
            if (logging) logger.logRaw(f);
            StrokeRecord rec;
            if (pipeline.onSample(f, rec)) {
                if (logging) logger.logStroke(rec);
                if (Serial) printStroke(rec);
            }
            ++nextN;
        }
        if (logging) logger.service();
    }

    const uint32_t now = millis();
    if (running && now - lastStatusMs >= 1000) {
        lastStatusMs = now;
        if (Serial) printStatus();
    }
    // LED: slow blink while calibrating, solid when running, fast blink on a hardware fault
    const bool led = hardwareFault ? (now / 63) & 1 : (!pipeline.calibrated() ? (now / 250) & 1 : running);
    digitalWrite(LED_BUILTIN, led ? HIGH : LOW);

    if (Serial.available()) {
        const int c = Serial.read();
        if ((c == 'q' || c == 'Q') && running) endSession();
    }
}
