// Native replay of the firmware core: raw_###.bin -> est.csv, compiled with the
// laptop's C++ compiler from the exact same lib/core sources the Teensy runs.
//
//   make -C firmware/test/native replay
//   ./replay raw_000.bin est_cpp.csv [--weights w.csv] [--subject subject.cfg]
//
// scripts/check_parity.py runs this and compares the result with the Python pipeline.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "FrameRing.h"
#include "LinearModel.h"
#include "ModelWeights.h"
#include "Pipeline.h"
#include "PipelineConfig.h"
#include "Subject.h"

static_assert(MW_D == PC_D, "ModelWeights.h and PipelineConfig.h disagree on D");

using namespace ree;

static constexpr size_t HEADER_BYTES = 32;
static constexpr size_t RECORD_BYTES = 4 + 4 + 2 * PC_N_CH + 1;

static bool readFile(const char* path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool loadWeights(const char* path, std::vector<double>& w) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        w.push_back(std::strtod(line.c_str(), nullptr));
    }
    return w.size() == static_cast<size_t>(PC_D);
}

static uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

static void writeRecord(FILE* fo, const StrokeRecord& r) {
    std::fprintf(fo, "%u,%u,%.17g,%.17g,%.17g,%d,%d,%.17g\n", r.n0, r.n1, r.T_s, r.P_w,
                 r.P_raw_w, r.valid ? 1 : 0, r.flagHigh ? 1 : 0, r.kcalTotal);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s raw.bin out.csv [--weights w.csv] [--subject subject.cfg]\n", argv[0]);
        return 2;
    }
    const char* weightsPath = nullptr;
    const char* subjectPath = nullptr;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--weights")) weightsPath = argv[i + 1];
        else if (!std::strcmp(argv[i], "--subject")) subjectPath = argv[i + 1];
    }

    std::vector<uint8_t> buf;
    if (!readFile(argv[1], buf) || buf.size() < HEADER_BYTES || std::memcmp(buf.data(), "REE1", 4) != 0) {
        std::fprintf(stderr, "not a raw_###.bin log: %s\n", argv[1]);
        return 1;
    }
    // header layout: magic[4] version@4 header_bytes@6 record_bytes@8 n_ch@10 n_imu@11 fs_hz@12
    //                 reserved@14 feature_hash@16 session@20 boot_ms@24 pad@28 (32 bytes)
    const uint8_t* h = buf.data();
    if (rd16(h + 6) != HEADER_BYTES || rd16(h + 8) != RECORD_BYTES || h[10] != PC_N_CH) {
        std::fprintf(stderr, "log layout (header %u, record %u bytes, %u channels) does not match this build\n",
                     rd16(h + 6), rd16(h + 8), h[10]);
        return 1;
    }
    if (rd32(h + 16) != PC_FEATURE_HASH)
        std::fprintf(stderr, "note: log feature hash 0x%08X != build 0x%08X\n", rd32(h + 16), PC_FEATURE_HASH);

    std::vector<double> w;
    if (weightsPath && !loadWeights(weightsPath, w)) {
        std::fprintf(stderr, "weights file must hold exactly %d values: %s\n", PC_D, weightsPath);
        return 1;
    }
    LinearModel model(weightsPath ? w.data() : MW_W, weightsPath ? false : MW_IS_PLACEHOLDER);

    Subject subj = defaultSubject();
    if (subjectPath) {
        std::vector<uint8_t> s;
        if (!readFile(subjectPath, s)) { std::fprintf(stderr, "cannot read %s\n", subjectPath); return 1; }
        parseSubjectCfg(reinterpret_cast<const char*>(s.data()), s.size(), subj);
    }

    std::vector<float> ringData(static_cast<size_t>(PC_RING_CAP) * PC_N_CH);
    std::vector<uint8_t> ringErr(PC_RING_CAP);
    FrameRing ring(ringData.data(), ringErr.data(), PC_RING_CAP);
    Pipeline pipe(ring, model);
    pipe.begin(subj);

    FILE* fo = std::fopen(argv[2], "w");
    if (!fo) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    std::fprintf(fo, "n0,n1,T_s,P_w,P_raw_w,valid,flag_high,kcal_total\n");

    const size_t nRec = (buf.size() - HEADER_BYTES) / RECORD_BYTES;
    RawFrame f{};
    uint32_t nPrev = 0, nLast = 0;
    size_t used = 0;
    for (size_t i = 0; i < nRec; ++i) {
        const uint8_t* p = buf.data() + HEADER_BYTES + i * RECORD_BYTES;
        f.n = rd32(p);
        if (i > 0 && f.n != nPrev + 1) break;   // preallocated tail / unclean close
        f.t_us = rd32(p + 4);
        std::memcpy(f.raw, p + 8, sizeof(f.raw));
        f.err = p[8 + sizeof(f.raw)];
        StrokeRecord r;
        if (pipe.onSample(f, r)) writeRecord(fo, r);
        nPrev = nLast = f.n;
        ++used;
    }
    const double kcal = used ? pipe.endSession(nLast + 1) : 0.0;
    std::fprintf(fo, "# end,%u,%.17g\n", used ? nLast + 1 : 0u, kcal);
    std::fclose(fo);
    std::printf("replayed %zu samples, %.6f kcal\n", used, kcal);
    return 0;
}
