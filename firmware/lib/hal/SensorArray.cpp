#include "SensorArray.h"

#include <string.h>

namespace ree {

bool SensorArray::add(ImuDriver* imu) {
    if (n_ >= PC_N_IMU || imu == nullptr) return false;
    imus_[n_++] = imu;
    return true;
}

uint8_t SensorArray::beginAll() {
    uint8_t failed = 0;
    for (int i = 0; i < n_; ++i) {
        if (!imus_[i]->begin()) failed |= static_cast<uint8_t>(1u << i);
    }
    return failed;
}

uint8_t SensorArray::readAll(int16_t raw[PC_N_CH]) {
    uint8_t mask = 0;
    for (int i = 0; i < n_; ++i) {
        int16_t v[6];
        const bool ok = imus_[i]->read(v) || imus_[i]->read(v);
        if (ok) {
            memcpy(last_[i], v, sizeof(v));
            consecutive_[i] = 0;
        } else {
            mask |= static_cast<uint8_t>(1u << i);
            ++failures_[i];
            if (++consecutive_[i] % RECOVER_EVERY == 0) imus_[i]->recover();
        }
        memcpy(raw + 6 * i, last_[i], sizeof(last_[i]));
    }
    return mask;
}

}  // namespace ree
