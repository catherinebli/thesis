#include "SdLogger.h"

#include <stdio.h>
#include <string.h>

namespace ree {

bool SdLogger::begin() { return sd_.begin(SdioConfig(FIFO_SDIO)); }

bool SdLogger::readSubject(Subject& s) {
    if (!sd_.exists("subject.cfg")) return false;
    FsFile f;
    if (!f.open("subject.cfg", O_RDONLY)) return false;
    char text[512];
    const int n = f.read(text, sizeof(text) - 1);
    f.close();
    if (n <= 0) return false;
    text[n] = '\0';
    parseSubjectCfg(text, static_cast<size_t>(n), s);
    return true;
}

bool SdLogger::openSession(uint32_t featureHash, uint32_t bootMs) {
    for (session_ = 0; session_ < 1000; ++session_) {
        snprintf(rawName_, sizeof(rawName_), "raw_%03u.bin", static_cast<unsigned>(session_));
        if (!sd_.exists(rawName_)) break;
    }
    if (session_ >= 1000) return false;
    snprintf(estName_, sizeof(estName_), "est_%03u.csv", static_cast<unsigned>(session_));

    if (!raw_.open(rawName_, O_RDWR | O_CREAT | O_TRUNC)) return false;
    if (!est_.open(estName_, O_RDWR | O_CREAT | O_TRUNC)) { raw_.close(); return false; }
    raw_.preAllocate(RAW_PREALLOC);
    est_.preAllocate(EST_PREALLOC);
    rawRing_.begin(&raw_);
    estRing_.begin(&est_);

    // header layout: magic[4] version@4 header_bytes@6 record_bytes@8 n_ch@10 n_imu@11
    //                fs_hz@12 reserved@14 feature_hash@16 session@20 boot_ms@24 pad@28
    uint8_t h[HEADER_BYTES] = {0};
    const uint16_t version = 1, hb = HEADER_BYTES, rb = RECORD_BYTES, fs = PC_FS_HZ;
    memcpy(h + 0, "REE1", 4);
    memcpy(h + 4, &version, 2);
    memcpy(h + 6, &hb, 2);
    memcpy(h + 8, &rb, 2);
    h[10] = PC_N_CH;
    h[11] = PC_N_IMU;
    memcpy(h + 12, &fs, 2);
    memcpy(h + 16, &featureHash, 4);
    memcpy(h + 20, &session_, 4);
    memcpy(h + 24, &bootMs, 4);
    rawRing_.write(h, sizeof(h));
    estRing_.write("n0,n1,T_s,P_w,P_raw_w,valid,flag_high,kcal_total\n");
    open_ = true;
    full_ = false;
    overruns_ = 0;
    return true;
}

bool SdLogger::logRaw(const RawFrame& f) {
    if (!open_ || full_) return false;
    if (rawRing_.bytesFree() < RECORD_BYTES) {
        ++overruns_;
        return false;
    }
    uint8_t rec[RECORD_BYTES];
    memcpy(rec + 0, &f.n, 4);                 // little-endian on the Teensy, as the reader expects
    memcpy(rec + 4, &f.t_us, 4);
    memcpy(rec + 8, f.raw, sizeof(f.raw));
    rec[8 + sizeof(f.raw)] = f.err;
    rawRing_.write(rec, sizeof(rec));
    return true;
}

bool SdLogger::logStroke(const StrokeRecord& r) {
    if (!open_) return false;
    char line[128];
    const int n = snprintf(line, sizeof(line), "%u,%u,%.4f,%.6f,%.6f,%d,%d,%.6f\n",
                           static_cast<unsigned>(r.n0), static_cast<unsigned>(r.n1), r.T_s, r.P_w,
                           r.P_raw_w, r.valid ? 1 : 0, r.flagHigh ? 1 : 0, r.kcalTotal);
    if (n <= 0 || estRing_.bytesFree() < static_cast<size_t>(n)) return false;
    estRing_.write(line, static_cast<size_t>(n));
    return true;
}

void SdLogger::service() {
    if (!open_) return;
    if (rawRing_.bytesUsed() >= 512 && !raw_.isBusy()) {
        if (raw_.curPosition() + 512 > RAW_PREALLOC) full_ = true;
        else rawRing_.writeOut(512);
    }
    if (estRing_.bytesUsed() >= 512 && !est_.isBusy()) estRing_.writeOut(512);
}

bool SdLogger::close(uint32_t nEnd, double kcalTotal) {
    if (!open_) return false;
    char line[64];
    const int n = snprintf(line, sizeof(line), "# end,%u,%.6f\n", static_cast<unsigned>(nEnd), kcalTotal);
    if (n > 0) estRing_.write(line, static_cast<size_t>(n));
    rawRing_.sync();
    raw_.truncate();
    raw_.close();
    estRing_.sync();
    est_.truncate();
    est_.close();
    open_ = false;
    return true;
}

}  // namespace ree
