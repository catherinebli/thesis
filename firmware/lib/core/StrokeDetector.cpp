#include "StrokeDetector.h"

#include <math.h>

namespace ree {

void StrokeDetector::reset() {
    candVal_ = -INFINITY;
    candN_ = -1;
    lastPeakN_ = -1;
    nRecent_ = 0;
}

double StrokeDetector::threshold() const {
    if (nRecent_ == 0) return PC_THR_MIN;
    double v[RECENT];
    for (int i = 0; i < nRecent_; ++i) v[i] = recent_[i];
    for (int i = 1; i < nRecent_; ++i) {      // insertion sort, at most 5 values
        double key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
        v[j + 1] = key;
    }
    const int m = nRecent_;
    const double med = (m % 2) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
    const double t = PC_THR_FRAC * med;
    return t > PC_THR_MIN ? t : PC_THR_MIN;
}

DetectEvent StrokeDetector::step(double s, uint32_t n, uint32_t& n0, uint32_t& n1) {
    const double thr = threshold();
    if (s > thr && s > candVal_) {
        candVal_ = s;
        candN_ = n;
        return DetectEvent::None;
    }
    if (candN_ >= 0 && s < PC_HYST * thr) {
        DetectEvent ev = DetectEvent::None;
        bool accept = true;
        if (lastPeakN_ < 0) {
            ev = DetectEvent::First;
            n0 = n1 = static_cast<uint32_t>(candN_);
        } else if (candN_ - lastPeakN_ >= PC_MIN_STROKE_N) {
            ev = DetectEvent::Stroke;
            n0 = static_cast<uint32_t>(lastPeakN_);
            n1 = static_cast<uint32_t>(candN_);
        } else {
            accept = false;                      // second bump inside one stroke
        }
        if (accept) {
            lastPeakN_ = candN_;
            if (nRecent_ < RECENT) {
                recent_[nRecent_++] = candVal_;
            } else {
                for (int i = 1; i < RECENT; ++i) recent_[i - 1] = recent_[i];
                recent_[RECENT - 1] = candVal_;
            }
        }
        candVal_ = -INFINITY;
        candN_ = -1;
        return ev;
    }
    return DetectEvent::None;
}

}  // namespace ree
