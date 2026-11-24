// Online drive-peak picker (mirror of python/rowing_ee/dsp/detector.py).
//
// Input: sign-corrected, filtered shank sagittal gyro. A candidate peak is
// tracked while the signal is above the adaptive threshold
//     thr = max(PC_THR_MIN, PC_THR_FRAC * median(last 5 accepted peaks))
// and confirmed once the signal drops below PC_HYST * thr. Peaks closer than
// PC_MIN_STROKE_N samples to the previous one are ignored (double bumps).
#pragma once
#include <stdint.h>

#include "Types.h"

namespace ree {

class StrokeDetector {
public:
    StrokeDetector() { reset(); }
    void reset();
    double threshold() const;

    // First  -> n0 = first confirmed peak of the session (n1 = n0)
    // Stroke -> one stroke spanning [n0, n1)
    DetectEvent step(double s, uint32_t n, uint32_t& n0, uint32_t& n1);

private:
    static constexpr int RECENT = 5;
    double candVal_;
    int64_t candN_;
    int64_t lastPeakN_;
    double recent_[RECENT];   // oldest first
    int nRecent_;
};

}  // namespace ree
