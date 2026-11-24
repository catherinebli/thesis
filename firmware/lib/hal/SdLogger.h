// SD logging on the Teensy 4.1 built-in slot (SDIO), following SdFat's
// TeensySdioLogger pattern: frames go into a RAM ring buffer, and service()
// moves 512-byte blocks to the card only when it is not busy, so a slow card
// write (which can stall 100+ ms) never blocks sampling.
//
//   raw_###.bin  32-byte header + one 69-byte record per 10 ms frame (see docs/06)
//   est_###.csv  one row per scored stroke, same columns as the Python output
#pragma once
#include <RingBuf.h>
#include <SdFat.h>

#include "Subject.h"
#include "Types.h"

namespace ree {

class SdLogger {
public:
    static constexpr size_t HEADER_BYTES = 32;
    static constexpr size_t RECORD_BYTES = 4 + 4 + 2 * PC_N_CH + 1;   // 69
    static constexpr size_t RAW_RING_BYTES = 64 * 512;                // ~4.7 s of frames
    static constexpr size_t EST_RING_BYTES = 8 * 512;
    static constexpr uint64_t RAW_PREALLOC = 64ull << 20;             // 64 MB, ~2.7 h
    static constexpr uint64_t EST_PREALLOC = 1ull << 20;

    bool begin();
    bool readSubject(Subject& s);                       // subject.cfg on the card root
    bool openSession(uint32_t featureHash, uint32_t bootMs);
    bool logRaw(const RawFrame& f);
    bool logStroke(const StrokeRecord& r);
    void service();                                     // call every loop()
    bool close(uint32_t nEnd, double kcalTotal);

    bool isOpen() const { return open_; }
    uint32_t session() const { return session_; }
    uint32_t overruns() const { return overruns_; }    // frames dropped because the ring was full
    const char* rawName() const { return rawName_; }

private:
    SdFs sd_;
    FsFile raw_, est_;
    RingBuf<FsFile, RAW_RING_BYTES> rawRing_;
    RingBuf<FsFile, EST_RING_BYTES> estRing_;
    char rawName_[16] = {0};
    char estName_[16] = {0};
    uint32_t session_ = 0;
    uint32_t overruns_ = 0;
    bool open_ = false;
    bool full_ = false;
};

}  // namespace ree
