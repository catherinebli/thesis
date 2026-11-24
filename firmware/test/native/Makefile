# Laptop builds of the firmware, no Teensy toolchain needed.
#
#   make replay      native replay of lib/core (used by scripts/check_parity.py)
#   make hal-check   syntax-check lib/hal and src/main.cpp against stub Arduino headers
#   make clean
#
# Regenerate ../../include/*.h with scripts/gen_headers.py after editing the config.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror
ROOT     := ../..
CORE     := $(ROOT)/lib/core
HAL      := $(ROOT)/lib/hal
INC      := -I$(ROOT)/include -I$(CORE)
CORE_SRC := $(wildcard $(CORE)/*.cpp)
DEPS     := $(CORE_SRC) $(wildcard $(CORE)/*.h) $(wildcard $(ROOT)/include/*.h) replay_main.cpp

replay: $(DEPS)
	$(CXX) $(CXXFLAGS) $(INC) replay_main.cpp $(CORE_SRC) -o $@ -lm

hal-check:
	$(CXX) -std=c++17 -fsyntax-only -Wall -Wextra -Werror -Istubs $(INC) -I$(HAL) \
	    $(wildcard $(HAL)/*.cpp) $(ROOT)/src/main.cpp

clean:
	rm -f replay

.PHONY: hal-check clean
