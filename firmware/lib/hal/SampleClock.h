// 100 Hz tick source. The ISR only counts ticks; all I2C work happens in loop().
// loop() compares ticks() with the next sample index it expects, so a slow
// iteration shows up as missed ticks (logged as ERR_GAP frames) instead of
// silently stretching time.
#pragma once
#include <Arduino.h>
#include <IntervalTimer.h>

namespace ree {

class SampleClock {
public:
    bool begin(uint32_t hz);
    void end() { timer_.end(); }
    uint32_t ticks() const { return ticks_; }   // 32-bit read is atomic on Cortex-M7

private:
    static void isr();
    static volatile uint32_t ticks_;
    IntervalTimer timer_;
};

}  // namespace ree
