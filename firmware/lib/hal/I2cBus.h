// One Teensy I2C port (Wire, Wire1 or Wire2) plus PCA9548 mux selection and
// stuck-bus recovery. Pins: Wire SDA 18 / SCL 19, Wire1 SDA 17 / SCL 16, Wire2 SDA 25 / SCL 24.
#pragma once
#include <Arduino.h>
#include <Wire.h>

namespace ree {

class I2cBus {
public:
    I2cBus(TwoWire& wire, uint8_t sdaPin, uint8_t sclPin) : wire_(wire), sda_(sdaPin), scl_(sclPin) {}

    void begin(uint32_t hz);
    bool write8(uint8_t addr, uint8_t reg, uint8_t val);
    // register burst read: write reg (repeated start), read n bytes
    bool readRegs(uint8_t addr, uint8_t reg, uint8_t* buf, uint8_t n);
    // Open one channel of a PCA9548 at muxAddr (0x70-0x77); muxAddr 0 = no mux.
    // Cached, so repeated reads of the same IMU cost no extra transactions.
    bool selectMux(uint8_t muxAddr, uint8_t channel);
    // A slave reset mid-transfer can hold SDA low forever: clock SCL up to 9 times,
    // send a STOP, then restart the peripheral.
    void recover();
    uint32_t errors() const { return errors_; }

private:
    TwoWire& wire_;
    uint8_t sda_, scl_;
    uint32_t hz_ = 400000;
    uint32_t errors_ = 0;
    int8_t muxSel_[8] = {-1, -1, -1, -1, -1, -1, -1, -1};   // open channel per mux 0x70+i
};

}  // namespace ree
