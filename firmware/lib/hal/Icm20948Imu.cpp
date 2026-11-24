#include "Icm20948Imu.h"

#include <Arduino.h>

#include "PipelineConfig.h"

namespace ree {

bool Icm20948Imu::setBank(uint8_t bank) {
    if (bank_ == static_cast<int8_t>(bank)) return true;
    if (!write(REG_BANK_SEL, static_cast<uint8_t>((bank << 4) & 0x30))) {
        bank_ = -1;
        return false;
    }
    bank_ = static_cast<int8_t>(bank);
    return true;
}

bool Icm20948Imu::begin() {
    if (!select()) return false;
    bank_ = -1;
    if (!setBank(0)) return false;
    uint8_t id = 0;
    if (!bus_.readRegs(addr_, REG_WHO_AM_I, &id, 1) || id != WHO_AM_I_VALUE) return false;

    write(REG_PWR_MGMT_1, 0x80);                 // device reset (bank returns to 0)
    delay(50);
    bank_ = 0;
    for (int i = 0; i < 10; ++i) {               // wait for the reset bit to clear
        uint8_t v = 0x80;
        if (bus_.readRegs(addr_, REG_PWR_MGMT_1, &v, 1) && !(v & 0x80)) break;
        delay(10);
    }
    bool ok = write(REG_PWR_MGMT_1, 0x01);       // wake (sleep bit 0), auto clock source
    delay(20);
    ok &= write(REG_PWR_MGMT_2, 0x00);           // accel and gyro, all axes on

    ok &= setBank(2);
    ok &= write(B2_GYRO_SMPLRT_DIV, static_cast<uint8_t>(PC_ICM_ODR_DIV & 0xFF));
    ok &= write(B2_GYRO_CONFIG_1,
                static_cast<uint8_t>((PC_ICM_DLPF_CFG << 3) | (PC_ICM_GYRO_FS_SEL << 1) | 0x01));
    ok &= write(B2_ACCEL_SMPLRT_DIV_1, static_cast<uint8_t>((PC_ICM_ODR_DIV >> 8) & 0x0F));
    ok &= write(B2_ACCEL_SMPLRT_DIV_2, static_cast<uint8_t>(PC_ICM_ODR_DIV & 0xFF));
    ok &= write(B2_ACCEL_CONFIG,
                static_cast<uint8_t>((PC_ICM_DLPF_CFG << 3) | (PC_ICM_ACCEL_FS_SEL << 1) | 0x01));
    ok &= setBank(0);
    delay(20);
    return ok;
}

bool Icm20948Imu::read(int16_t out[6]) {
    if (!select()) return false;
    if (!setBank(0)) return false;
    uint8_t b[12];
    if (!bus_.readRegs(addr_, REG_ACCEL_XOUT_H, b, 12)) return false;
    for (int k = 0; k < 6; ++k)
        out[k] = static_cast<int16_t>(static_cast<uint16_t>(b[2 * k] << 8) | b[2 * k + 1]);
    return true;
}

bool Icm20948Imu::recover() {
    bus_.recover();
    return begin();
}

}  // namespace ree
