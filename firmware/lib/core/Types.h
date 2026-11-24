// Shared plain-data types for the portable core. No Arduino headers in lib/core:
// these files also compile on a laptop (firmware/test/native) for parity tests.
#pragma once
#include <stdint.h>

#include "PipelineConfig.h"

namespace ree {

// err bits: bit i = IMU i read failed and its last good value was held;
// ERR_GAP = the 10 ms tick was missed and the whole frame is a held copy.
constexpr uint8_t ERR_GAP = 0x80;

struct RawFrame {
    uint32_t n;               // sample index, +1 per tick
    uint32_t t_us;            // micros() when read (diagnostics only; timing uses n)
    int16_t raw[PC_N_CH];     // ax ay az gx gy gz per IMU, raw counts, sensor frame
    uint8_t err;
};

struct StrokeRecord {
    uint32_t n0, n1;          // stroke spans samples [n0, n1)
    double T_s;               // (n1 - n0) / fs
    double P_w;               // power billed for the stroke (held value if invalid)
    double P_raw_w;           // model output before floor and cap (NaN if not scored)
    bool valid;
    bool flagHigh;            // model output exceeded PC_P_MAX_W and was capped
    double kcalTotal;         // cumulative kcal at the end of the stroke
};

enum class DetectEvent : uint8_t { None, First, Stroke };

}  // namespace ree
