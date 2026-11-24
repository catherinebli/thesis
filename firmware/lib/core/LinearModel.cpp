#include "LinearModel.h"

namespace ree {

double LinearModel::raw(const double x[PC_D]) const {
    double acc = 0.0;
    for (int i = 0; i < PC_D; ++i) acc += w_[i] * x[i];
    return acc;
}

double LinearModel::predict(const double x[PC_D], double pRest, double& pRaw, bool& flagHigh) const {
    pRaw = raw(x);
    flagHigh = pRaw > PC_P_MAX_W;
    double p = pRaw < PC_P_MAX_W ? pRaw : PC_P_MAX_W;
    return p > pRest ? p : pRest;
}

}  // namespace ree
