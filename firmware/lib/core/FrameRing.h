// Last PC_RING_CAP filtered frames, indexed by absolute sample number n
// (mirror of python/rowing_ee/dsp/ring.py). Storage is passed in so the
// firmware can place it in RAM2 (DMAMEM) without the core knowing about Teensy.
#pragma once
#include <stdint.h>

#include "PipelineConfig.h"

namespace ree {

class FrameRing {
public:
    FrameRing(float* data, uint8_t* err, uint32_t capacity)
        : data_(data), err_(err), cap_(capacity), newest_(0), count_(0) {}

    void clear() { count_ = 0; newest_ = 0; }
    void push(uint32_t n, const double y[PC_N_CH], uint8_t err);

    // true if every sample in [n0, n1) is still buffered
    bool contains(uint32_t n0, uint32_t n1) const;
    float at(uint32_t n, int ch) const { return data_[(n % cap_) * PC_N_CH + ch]; }
    uint8_t errAt(uint32_t n) const { return err_[n % cap_]; }
    uint32_t newest() const { return newest_; }
    uint32_t count() const { return count_; }

private:
    float* data_;
    uint8_t* err_;
    uint32_t cap_;
    uint32_t newest_;
    uint32_t count_;
};

}  // namespace ree
