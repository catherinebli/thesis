// One stroke -> x[PC_D] (mirror of python/rowing_ee/dsp/features.py).
//   x = [1, mass_kg, height_m, T_s, ch0 bins 0..B-1, ch1 bins 0..B-1, ...]
// Bin k of an L-sample stroke averages samples [floor(kL/B), floor((k+1)L/B)).
#pragma once
#include <stdint.h>

#include "FrameRing.h"
#include "Subject.h"

namespace ree {

class Featurizer {
public:
    // Caller guarantees ring.contains(n0, n1) and n1 - n0 >= PC_BINS.
    void extract(const FrameRing& ring, uint32_t n0, uint32_t n1, const Subject& s,
                 double x[PC_D]) const;
};

}  // namespace ree
