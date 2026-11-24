#pragma once
#include <stddef.h>
#include <stdint.h>
template <class F, size_t Size>
class RingBuf {
public:
    void begin(F* file);
    size_t bytesFree() const;
    size_t bytesUsed() const;
    size_t write(const void* buf, size_t count);
    size_t write(const char* str);
    size_t writeOut(size_t count);
    bool sync();
};
