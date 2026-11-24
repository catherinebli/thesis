#pragma once
#include "Arduino.h"
class TwoWire {
public:
    void begin();
    void end();
    void setClock(uint32_t frequency);
    void beginTransmission(uint8_t address);
    uint8_t endTransmission(uint8_t sendStop);
    uint8_t endTransmission(void);
    uint8_t requestFrom(uint8_t address, uint8_t quantity);
    size_t write(uint8_t data);
    int available();
    int read();
};
extern TwoWire Wire;
extern TwoWire Wire1;
extern TwoWire Wire2;
