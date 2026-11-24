#include "FilterBank.h"

namespace ree {

void FilterBank::reset(const double x0[PC_N_CH]) {
    for (int c = 0; c < PC_N_CH; ++c) {
        double u = x0[c];
        for (int s = 0; s < PC_N_SOS; ++s) {
            const double* q = PC_SOS[s];  // b0 b1 b2 a0 a1 a2
            const double y = u * (q[0] + q[1] + q[2]) / (1.0 + q[4] + q[5]);
            z_[s][0][c] = (q[1] + q[2]) * u - (q[4] + q[5]) * y;
            z_[s][1][c] = q[2] * u - q[5] * y;
            u = y;
        }
    }
}

void FilterBank::step(const double x[PC_N_CH], double y[PC_N_CH]) {
    for (int c = 0; c < PC_N_CH; ++c) {
        double u = x[c];
        for (int s = 0; s < PC_N_SOS; ++s) {
            const double* q = PC_SOS[s];
            const double out = q[0] * u + z_[s][0][c];
            z_[s][0][c] = q[1] * u - q[4] * out + z_[s][1][c];
            z_[s][1][c] = q[2] * u - q[5] * out;
            u = out;
        }
        y[c] = u;
    }
}

}  // namespace ree
