// The five IMUs in config order. A failed read is retried once; if it still
// fails, the IMU's last good values are held and its bit is set in the error
// mask, so the frame stays complete and the pipeline can score around it.
#pragma once
#include <stdint.h>

#include "ImuDriver.h"
#include "PipelineConfig.h"

namespace ree {

class SensorArray {
public:
    bool add(ImuDriver* imu);
    uint8_t beginAll();                        // bitmask of IMUs that failed to start
    uint8_t readAll(int16_t raw[PC_N_CH]);     // error mask for this frame
    uint32_t failures(int i) const { return failures_[i]; }
    int size() const { return n_; }
    ImuDriver* imu(int i) { return imus_[i]; }

private:
    static constexpr uint16_t RECOVER_EVERY = 100;   // consecutive failures before a re-init
    ImuDriver* imus_[PC_N_IMU] = {};
    int n_ = 0;
    int16_t last_[PC_N_IMU][6] = {};
    uint16_t consecutive_[PC_N_IMU] = {};
    uint32_t failures_[PC_N_IMU] = {};
};

}  // namespace ree
