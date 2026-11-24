// Participant anthropometrics and resting power (mirror of python/rowing_ee/subject.py).
#pragma once
#include <stddef.h>

namespace ree {

constexpr double KCAL_PER_DAY_TO_W = 4184.0 / 86400.0;  // 0.0484259...

struct Subject {
    char id[16];
    double massKg;
    double heightM;
    double age;
    char sex;  // 'M' or 'F'

    // Mifflin-St Jeor basal metabolic rate in W
    double bmrWatts() const;
    // floor for every estimate, and the power billed during pauses
    double restWatts(double kRest) const { return bmrWatts() * kRest; }
};

// Subject from PipelineConfig.h defaults.
Subject defaultSubject();

// Parse the key=value subject.cfg read from the SD card (id, mass_kg, height_m, age, sex).
// Unknown keys and malformed lines are ignored; missing keys keep the value already in `s`.
void parseSubjectCfg(const char* text, size_t len, Subject& s);

}  // namespace ree
