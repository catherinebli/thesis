// Watts per stroke -> joules -> kcal (mirror of python/rowing_ee/energy.py).
// Time is counted in samples; strokes tile time from one drive peak to the next:
//   E = sum_n P_n * T_n + P_rest * T_rest,   kcal = E / 4184
#pragma once
#include <stdint.h>

namespace ree {

class EnergyMeter {
public:
    enum Mode : uint8_t { Rest = 0, Rowing = 1 };

    void configure(double fs, uint32_t restTimeoutN, double pRest) {
        fs_ = fs; restTimeoutN_ = restTimeoutN; pRest_ = pRest;
    }
    void begin(uint32_t n);
    // first catch after the start or a pause: bill the gap at resting power
    void startBout(uint32_t n);
    // bill [nAcc, n1) at the stroke power; invalid strokes hold the last estimate
    double addStroke(uint32_t n0, uint32_t n1, double p, bool valid);
    void tick(uint32_t n);

    Mode mode() const { return mode_; }
    double liveJoules(uint32_t n) const;
    double liveKcal(uint32_t n) const { return liveJoules(n) / 4184.0; }
    double end(uint32_t n);   // returns total kcal

private:
    double fs_ = 100.0;
    uint32_t restTimeoutN_ = 800;
    double pRest_ = 0.0;
    double E_ = 0.0;
    uint32_t nAcc_ = 0;
    Mode mode_ = Rest;
    double pLast_ = 0.0;
    uint32_t nLastPeak_ = 0;
};

}  // namespace ree
