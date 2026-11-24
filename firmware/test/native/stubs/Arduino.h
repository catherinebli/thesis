// Minimal stand-ins for the Teensy core, only so `make hal-check` can
// syntax-check lib/hal and src/main.cpp on a laptop. Signatures follow
// PaulStoffregen/cores (teensy4). Not used by the real build.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#define DMAMEM __attribute__((section(".dmabuffers"), used))
#define HIGH 1
#define LOW 0
#define INPUT_PULLUP 2
#define OUTPUT 1
#define OUTPUT_OPENDRAIN 4
#define LED_BUILTIN 13
void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t val);
uint8_t digitalRead(uint8_t pin);
void delay(uint32_t ms);
void delayMicroseconds(uint32_t us);
uint32_t millis();
uint32_t micros();
class Print {
public:
    size_t write(uint8_t);
    size_t write(const char*);
    size_t write(const uint8_t*, size_t);
    size_t print(const char*);
    size_t println(const char*);
    int printf(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
};
class usb_serial_class : public Print {
public:
    void begin(long);
    int available();
    int read();
    explicit operator bool();
};
extern usb_serial_class Serial;
