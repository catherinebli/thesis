#include "FrameRing.h"

namespace ree {

void FrameRing::push(uint32_t n, const double y[PC_N_CH], uint8_t err) {
    float* row = data_ + (n % cap_) * PC_N_CH;
    for (int c = 0; c < PC_N_CH; ++c) row[c] = static_cast<float>(y[c]);
    err_[n % cap_] = err;
    newest_ = n;
    if (count_ < cap_) ++count_;
}

bool FrameRing::contains(uint32_t n0, uint32_t n1) const {
    if (count_ == 0 || n0 > n1) return false;
    const uint32_t oldest = newest_ - count_ + 1;
    return n0 >= oldest && n1 <= newest_ + 1;
}

}  // namespace ree
