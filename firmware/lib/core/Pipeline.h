// Orchestrates one frame at a time (mirror of python/rowing_ee/pipeline.py):
//   calibrate -> filter -> ring buffer -> detector -> (stroke) features -> model -> energy
#pragma once
#include <stdint.h>

#include "Calibration.h"
#include "EnergyMeter.h"
#include "Featurizer.h"
#include "FilterBank.h"
#include "FrameRing.h"
#include "LinearModel.h"
#include "StrokeDetector.h"
#include "Subject.h"
#include "Types.h"

namespace ree {

class Pipeline {
public:
    Pipeline(FrameRing& ring, const LinearModel& model) : ring_(ring), model_(model) {}

    // Reset everything for a new session (calibration starts again).
    void begin(const Subject& s);
    // Feed one frame. Returns true and fills `out` when a stroke was scored.
    bool onSample(const RawFrame& f, StrokeRecord& out);

    double liveKcal(uint32_t n) const { return energy_.liveKcal(n); }
    double endSession(uint32_t nEnd) { return energy_.end(nEnd); }   // nEnd exclusive

    bool calibrated() const { return cal_.done(); }
    int64_t calibratedAt() const { return calN_; }
    double restWatts() const { return pRest_; }
    const Calibration& calibration() const { return cal_; }
    const Subject& subject() const { return subj_; }
    EnergyMeter::Mode mode() const { return energy_.mode(); }

private:
    bool onStroke(uint32_t n0, uint32_t n1, StrokeRecord& out);

    FrameRing& ring_;
    const LinearModel& model_;
    Subject subj_{};
    Calibration cal_;
    FilterBank fb_;
    StrokeDetector det_;
    Featurizer feat_;
    EnergyMeter energy_;
    bool started_ = false;
    bool filterInit_ = false;
    int64_t calN_ = -1;
    double pRest_ = 0.0;
    double x_[PC_D];
};

}  // namespace ree
