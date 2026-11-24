#include "Featurizer.h"

namespace ree {

void Featurizer::extract(const FrameRing& ring, uint32_t n0, uint32_t n1, const Subject& s,
                         double x[PC_D]) const {
    const uint32_t L = n1 - n0;
    x[0] = 1.0;
    x[1] = s.massKg;
    x[2] = s.heightM;
    x[3] = static_cast<double>(L) / PC_FS_HZ;
    for (int c = 0; c < PC_N_CH; ++c) {
        for (uint32_t k = 0; k < PC_BINS; ++k) {
            const uint32_t a = n0 + (k * L) / PC_BINS;
            const uint32_t b = n0 + ((k + 1) * L) / PC_BINS;
            double acc = 0.0;
            for (uint32_t n = a; n < b; ++n) acc += static_cast<double>(ring.at(n, c));
            x[4 + c * PC_BINS + k] = acc / static_cast<double>(b - a);
        }
    }
}

}  // namespace ree
