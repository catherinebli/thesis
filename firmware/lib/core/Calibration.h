// Counts -> SI units, gyro bias, mount rotation (mirror of python/rowing_ee/dsp/calibration.py).
#pragma once
#include <stdint.h>

#include "PipelineConfig.h"

namespace ree {

class Calibration {
public:
    Calibration() { reset(); }
    void reset();

    // Accumulate one frame of the power-on still window. Returns true when the
    // gyro bias has been set. A window whose gyro std exceeds
    // PC_MAX_STILL_GYRO_STD on any axis is discarded and the next one is tried.
    bool addStill(const int16_t raw[PC_N_CH]);
    bool done() const { return done_; }
    int windowsTried() const { return windows_; }
    const double* bias() const { return bias_; }   // rad/s, sensor frame, 3 per IMU

    // raw counts -> segment-frame m/s^2 and rad/s (gyro bias removed)
    void apply(const int16_t raw[PC_N_CH], double out[PC_N_CH]) const;

private:
    double sum_[PC_N_IMU * 3];
    double sumsq_[PC_N_IMU * 3];
    double bias_[PC_N_IMU * 3];
    uint32_t count_;
    int windows_;
    bool done_;
};

}  // namespace ree
