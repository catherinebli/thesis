#include "Calibration.h"

#include <math.h>
#include <string.h>

namespace ree {

void Calibration::reset() {
    memset(sum_, 0, sizeof(sum_));
    memset(sumsq_, 0, sizeof(sumsq_));
    memset(bias_, 0, sizeof(bias_));
    count_ = 0;
    windows_ = 0;
    done_ = false;
}

bool Calibration::addStill(const int16_t raw[PC_N_CH]) {
    for (int i = 0; i < PC_N_IMU; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double g = static_cast<double>(raw[6 * i + 3 + k]) * PC_GYRO_SCALE;
            sum_[3 * i + k] += g;
            sumsq_[3 * i + k] += g * g;
        }
    }
    ++count_;
    if (count_ < PC_STILL_N) return false;

    ++windows_;
    double maxStd = 0.0;
    double mean[PC_N_IMU * 3];
    for (int j = 0; j < PC_N_IMU * 3; ++j) {
        mean[j] = sum_[j] / count_;
        double var = sumsq_[j] / count_ - mean[j] * mean[j];
        double sd = sqrt(var > 0.0 ? var : 0.0);
        if (sd > maxStd) maxStd = sd;
    }
    if (maxStd <= PC_MAX_STILL_GYRO_STD) {
        memcpy(bias_, mean, sizeof(bias_));
        done_ = true;
        return true;
    }
    memset(sum_, 0, sizeof(sum_));      // person moved: try the next window
    memset(sumsq_, 0, sizeof(sumsq_));
    count_ = 0;
    return false;
}

void Calibration::apply(const int16_t raw[PC_N_CH], double out[PC_N_CH]) const {
    for (int i = 0; i < PC_N_IMU; ++i) {
        double a[3], g[3];
        for (int k = 0; k < 3; ++k) {
            a[k] = static_cast<double>(raw[6 * i + k]) * PC_ACCEL_SCALE;
            g[k] = static_cast<double>(raw[6 * i + 3 + k]) * PC_GYRO_SCALE - bias_[3 * i + k];
        }
        for (int r = 0; r < 3; ++r) {
            double sa = 0.0, sg = 0.0;
            for (int c = 0; c < 3; ++c) {
                sa += PC_MOUNT[i][r][c] * a[c];
                sg += PC_MOUNT[i][r][c] * g[c];
            }
            out[6 * i + r] = sa;
            out[6 * i + 3 + r] = sg;
        }
    }
}

}  // namespace ree
