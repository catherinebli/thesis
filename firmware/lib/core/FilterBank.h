// Causal Butterworth low-pass on every channel (mirror of python/rowing_ee/dsp/filters.py).
// Direct form II transposed second-order sections, the same recurrence as scipy's sosfilt.
#pragma once
#include "PipelineConfig.h"

namespace ree {

class FilterBank {
public:
    // Steady-state start for a constant input x0, so there is no start-up transient
    // (gravity on the accel channels would otherwise ramp up from zero).
    void reset(const double x0[PC_N_CH]);
    void step(const double x[PC_N_CH], double y[PC_N_CH]);

private:
    double z_[PC_N_SOS][2][PC_N_CH];
};

}  // namespace ree
