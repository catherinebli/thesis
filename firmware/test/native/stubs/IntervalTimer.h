#pragma once
#include <stdint.h>
class IntervalTimer {
public:
    typedef void (*callback_t)();
    template <typename period_t> bool begin(callback_t funct, period_t period);
    void end();
};
