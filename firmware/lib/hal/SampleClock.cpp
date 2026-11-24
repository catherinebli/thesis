#include "SampleClock.h"

namespace ree {

volatile uint32_t SampleClock::ticks_ = 0;

void SampleClock::isr() { ticks_ = ticks_ + 1; }

bool SampleClock::begin(uint32_t hz) {
    ticks_ = 0;
    return timer_.begin(isr, 1000000.0f / static_cast<float>(hz));
}

}  // namespace ree
