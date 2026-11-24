// Interface every IMU chip driver implements. Swapping chips means writing one
// new subclass; nothing in lib/core changes (it only sees raw counts).
#pragma once
#include <stdint.h>

namespace ree {

class ImuDriver {
public:
    virtual ~ImuDriver() {}
    virtual bool begin() = 0;                    // probe, reset, configure
    virtual bool read(int16_t out[6]) = 0;       // ax ay az gx gy gz raw counts, sensor frame
    virtual bool recover() { return begin(); }   // called after repeated read failures
    virtual const char* name() const = 0;
};

}  // namespace ree
