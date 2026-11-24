#include "Pipeline.h"

#include <math.h>

namespace ree {

void Pipeline::begin(const Subject& s) {
    subj_ = s;
    pRest_ = s.restWatts(PC_K_REST);
    cal_.reset();
    det_.reset();
    ring_.clear();
    energy_.configure(static_cast<double>(PC_FS_HZ), PC_REST_TIMEOUT_N, pRest_);
    started_ = false;
    filterInit_ = false;
    calN_ = -1;
}

bool Pipeline::onSample(const RawFrame& f, StrokeRecord& out) {
    if (!started_) {
        energy_.begin(f.n);
        started_ = true;
    }
    if (!cal_.done()) {
        if (cal_.addStill(f.raw)) calN_ = f.n;
        return false;
    }
    double x[PC_N_CH], y[PC_N_CH];
    cal_.apply(f.raw, x);
    if (!filterInit_) {
        fb_.reset(x);
        filterInit_ = true;
    }
    fb_.step(x, y);
    ring_.push(f.n, y, f.err);

    uint32_t n0 = 0, n1 = 0;
    bool produced = false;
    const DetectEvent ev = det_.step(PC_DETECT_SIGN * y[PC_DETECT_CH], f.n, n0, n1);
    if (ev == DetectEvent::First) {
        energy_.startBout(n0);
    } else if (ev == DetectEvent::Stroke) {
        produced = onStroke(n0, n1, out);
    }
    energy_.tick(f.n);
    return produced;
}

bool Pipeline::onStroke(uint32_t n0, uint32_t n1, StrokeRecord& out) {
    const uint32_t L = n1 - n0;
    if (energy_.mode() == EnergyMeter::Rest || L > PC_MAX_STROKE_N) {
        energy_.startBout(n1);              // first stroke after a pause: no estimate
        return false;
    }
    bool valid = ring_.contains(n0, n1);
    if (valid) {
        uint32_t nErr = 0;
        for (uint32_t n = n0; n < n1; ++n) nErr += ring_.errAt(n) ? 1u : 0u;
        valid = static_cast<double>(nErr) / static_cast<double>(L) <= PC_MAX_ERR_FRAC;
    }
    double pRaw = NAN, p = NAN;
    bool flag = false;
    if (valid) {
        feat_.extract(ring_, n0, n1, subj_, x_);
        p = model_.predict(x_, pRest_, pRaw, flag);
    }
    const double used = energy_.addStroke(n0, n1, p, valid);
    out.n0 = n0;
    out.n1 = n1;
    out.T_s = static_cast<double>(L) / PC_FS_HZ;
    out.P_w = used;
    out.P_raw_w = pRaw;
    out.valid = valid;
    out.flagHigh = flag;
    out.kcalTotal = energy_.liveKcal(n1);
    return true;
}

}  // namespace ree
