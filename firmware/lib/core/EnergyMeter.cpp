#include "EnergyMeter.h"

namespace ree {

void EnergyMeter::begin(uint32_t n) {
    E_ = 0.0;
    nAcc_ = n;
    mode_ = Rest;
    pLast_ = pRest_;
    nLastPeak_ = n;
}

void EnergyMeter::startBout(uint32_t n) {
    E_ += pRest_ * static_cast<double>(n - nAcc_) / fs_;
    nAcc_ = n;
    mode_ = Rowing;
    nLastPeak_ = n;
}

double EnergyMeter::addStroke(uint32_t /*n0*/, uint32_t n1, double p, bool valid) {
    if (!valid) p = pLast_;
    E_ += p * static_cast<double>(n1 - nAcc_) / fs_;
    nAcc_ = n1;
    if (valid) pLast_ = p;
    nLastPeak_ = n1;
    return p;
}

void EnergyMeter::tick(uint32_t n) {
    if (mode_ == Rowing && (n - nLastPeak_) > restTimeoutN_) mode_ = Rest;
}

double EnergyMeter::liveJoules(uint32_t n) const {
    const double p = (mode_ == Rowing) ? pLast_ : pRest_;
    return E_ + p * static_cast<double>(n - nAcc_) / fs_;
}

double EnergyMeter::end(uint32_t n) {
    const double p = (mode_ == Rowing) ? pLast_ : pRest_;
    E_ += p * static_cast<double>(n - nAcc_) / fs_;
    nAcc_ = n;
    return E_ / 4184.0;
}

}  // namespace ree
