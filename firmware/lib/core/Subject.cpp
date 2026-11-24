#include "Subject.h"

#include <stdlib.h>
#include <string.h>

#include "PipelineConfig.h"

namespace ree {

double Subject::bmrWatts() const {
    const double s = (sex == 'F' || sex == 'f') ? -161.0 : 5.0;
    return (10.0 * massKg + 625.0 * heightM - 5.0 * age + s) * KCAL_PER_DAY_TO_W;
}

Subject defaultSubject() {
    Subject s{};
    strncpy(s.id, PC_SUBJ_ID, sizeof(s.id) - 1);
    s.massKg = PC_SUBJ_MASS_KG;
    s.heightM = PC_SUBJ_HEIGHT_M;
    s.age = PC_SUBJ_AGE;
    s.sex = PC_SUBJ_SEX;
    return s;
}

static void trim(char*& b, char*& e) {
    while (b < e && (*b == ' ' || *b == '\t' || *b == '\r')) ++b;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
}

void parseSubjectCfg(const char* text, size_t len, Subject& s) {
    char line[96];
    size_t i = 0;
    while (i < len) {
        size_t j = i;
        while (j < len && text[j] != '\n') ++j;
        size_t n = j - i < sizeof(line) - 1 ? j - i : sizeof(line) - 1;
        memcpy(line, text + i, n);
        line[n] = '\0';
        i = j + 1;

        char* hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char* eq = strchr(line, '=');
        if (!eq) continue;
        char *kb = line, *ke = eq, *vb = eq + 1, *ve = line + strlen(line);
        trim(kb, ke);
        trim(vb, ve);
        *ke = '\0';
        *ve = '\0';
        if (vb == ve) continue;
        char* endp = nullptr;
        if (strcmp(kb, "mass_kg") == 0 || strcmp(kb, "height_m") == 0 || strcmp(kb, "age") == 0) {
            double v = strtod(vb, &endp);
            if (endp == vb) continue;
            if (kb[0] == 'm') s.massKg = v;
            else if (kb[0] == 'h') s.heightM = v;
            else s.age = v;
        } else if (strcmp(kb, "id") == 0) {
            memset(s.id, 0, sizeof(s.id));
            strncpy(s.id, vb, sizeof(s.id) - 1);
        } else if (strcmp(kb, "sex") == 0) {
            s.sex = (vb[0] == 'f' || vb[0] == 'F') ? 'F' : 'M';
        }
    }
}

}  // namespace ree
