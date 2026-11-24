#pragma once
#include <stdint.h>
#include <stddef.h>
#define O_RDONLY 0x00
#define O_RDWR 0x02
#define O_CREAT 0x10
#define O_TRUNC 0x20
#define FIFO_SDIO 0
class SdioConfig { public: explicit SdioConfig(uint8_t opt); };
class FsFile {
public:
    bool open(const char* path, int oflag);
    int read(void* buf, size_t count);
    bool preAllocate(uint64_t length);
    bool isBusy();
    uint64_t curPosition() const;
    bool truncate();
    bool close();
};
class SdFs {
public:
    bool begin(SdioConfig config);
    bool exists(const char* path);
};
