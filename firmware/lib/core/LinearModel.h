// P = w . x (mirror of python/rowing_ee/model/linear_model.py).
// Weights come from ModelWeights.h (generated) or, in the native replay tool, from a CSV.
#pragma once
#include "PipelineConfig.h"

namespace ree {

class LinearModel {
public:
    LinearModel(const double* w, bool placeholder) : w_(w), placeholder_(placeholder) {}
    void setWeights(const double* w, bool placeholder) { w_ = w; placeholder_ = placeholder; }
    bool isPlaceholder() const { return placeholder_; }

    double raw(const double x[PC_D]) const;
    // Model output capped at PC_P_MAX_W (flagHigh set) and floored at pRest.
    double predict(const double x[PC_D], double pRest, double& pRaw, bool& flagHigh) const;

private:
    const double* w_;
    bool placeholder_;
};

}  // namespace ree
