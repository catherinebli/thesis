// Minimal register-level driver for the Adafruit ICM-20948 breakout (accel + gyro only).
// The magnetometer and DMP are not used: raw polling, no heading (steel erg rail).
// Register map cross-checked against Adafruit_ICM20X and the SparkFun ICM-20948 library.
#pragma once
#include <stdint.h>

#include "I2cBus.h"
#include "ImuDriver.h"

namespace ree {

class Icm20948Imu : public ImuDriver {
public:
    // bank 0
    static constexpr uint8_t REG_WHO_AM_I = 0x00;      // reads 0xEA
    static constexpr uint8_t REG_PWR_MGMT_1 = 0x06;    // bit7 reset, bit6 sleep, [2:0] clock
    static constexpr uint8_t REG_PWR_MGMT_2 = 0x07;    // 0 = all accel and gyro axes on
    static constexpr uint8_t REG_ACCEL_XOUT_H = 0x2D;  // 12 bytes: accel xyz then gyro xyz, big-endian
    static constexpr uint8_t REG_BANK_SEL = 0x7F;      // bank in bits [5:4]
    // bank 2
    static constexpr uint8_t B2_GYRO_SMPLRT_DIV = 0x00;
    static constexpr uint8_t B2_GYRO_CONFIG_1 = 0x01;     // [5:3] DLPFCFG, [2:1] FS_SEL, [0] FCHOICE
    static constexpr uint8_t B2_ACCEL_SMPLRT_DIV_1 = 0x10;
    static constexpr uint8_t B2_ACCEL_SMPLRT_DIV_2 = 0x11;
    static constexpr uint8_t B2_ACCEL_CONFIG = 0x14;      // [5:3] DLPFCFG, [2:1] FS_SEL, [0] FCHOICE
    static constexpr uint8_t WHO_AM_I_VALUE = 0xEA;

    Icm20948Imu(I2cBus& bus, uint8_t addr, uint8_t muxAddr, uint8_t muxChannel, const char* name)
        : bus_(bus), addr_(addr), mux_(muxAddr), ch_(muxChannel), name_(name) {}

    bool begin() override;
    bool read(int16_t out[6]) override;
    bool recover() override;
    const char* name() const override { return name_; }

private:
    bool select() { return bus_.selectMux(mux_, ch_); }
    bool setBank(uint8_t bank);
    bool write(uint8_t reg, uint8_t v) { return bus_.write8(addr_, reg, v); }

    I2cBus& bus_;
    uint8_t addr_, mux_, ch_;
    const char* name_;
    int8_t bank_ = -1;
};

}  // namespace ree
