#include "I2cBus.h"

namespace ree {

void I2cBus::begin(uint32_t hz) {
    hz_ = hz;
    wire_.begin();
    wire_.setClock(hz_);
    for (auto& m : muxSel_) m = -1;
}

bool I2cBus::write8(uint8_t addr, uint8_t reg, uint8_t val) {
    wire_.beginTransmission(addr);
    wire_.write(reg);
    wire_.write(val);
    if (wire_.endTransmission() != 0) {
        ++errors_;
        return false;
    }
    return true;
}

bool I2cBus::readRegs(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t n) {
    wire_.beginTransmission(addr);
    wire_.write(reg);
    if (wire_.endTransmission(0) != 0) {        // 0 = repeated start, keep the bus
        ++errors_;
        return false;
    }
    if (wire_.requestFrom(addr, n) != n) {
        while (wire_.available()) wire_.read();
        ++errors_;
        return false;
    }
    for (uint8_t i = 0; i < n; ++i) buf[i] = static_cast<uint8_t>(wire_.read());
    return true;
}

bool I2cBus::selectMux(uint8_t muxAddr, uint8_t channel) {
    if (muxAddr == 0) return true;
    const int idx = muxAddr - 0x70;
    if (idx < 0 || idx > 7 || channel > 7) return false;
    if (muxSel_[idx] == static_cast<int8_t>(channel)) return true;
    // two muxes on one bus: close the other one first so addresses cannot collide
    for (int j = 0; j < 8; ++j) {
        if (j != idx && muxSel_[j] >= 0) {
            wire_.beginTransmission(static_cast<uint8_t>(0x70 + j));
            wire_.write(0);
            wire_.endTransmission();
            muxSel_[j] = -1;
        }
    }
    wire_.beginTransmission(muxAddr);
    wire_.write(static_cast<uint8_t>(1u << channel));
    if (wire_.endTransmission() != 0) {
        ++errors_;
        muxSel_[idx] = -1;
        return false;
    }
    muxSel_[idx] = static_cast<int8_t>(channel);
    return true;
}

void I2cBus::recover() {
    wire_.end();
    pinMode(sda_, INPUT_PULLUP);
    pinMode(scl_, OUTPUT_OPENDRAIN);
    digitalWrite(scl_, HIGH);
    for (int i = 0; i < 9 && digitalRead(sda_) == LOW; ++i) {
        digitalWrite(scl_, LOW);
        delayMicroseconds(5);
        digitalWrite(scl_, HIGH);
        delayMicroseconds(5);
    }
    pinMode(sda_, OUTPUT_OPENDRAIN);           // STOP: SDA rises while SCL is high
    digitalWrite(sda_, LOW);
    delayMicroseconds(5);
    digitalWrite(scl_, HIGH);
    delayMicroseconds(5);
    digitalWrite(sda_, HIGH);
    delayMicroseconds(5);
    begin(hz_);                                // re-attaches the pins to the I2C peripheral
}

}  // namespace ree
