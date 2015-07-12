ifeq ($(shell uname -s),Darwin)
	CC ?= clang
	LDLIBS += -lcurl -framework OpenCL
else
	CC ?= gcc
	LDLIBS += -lOpenCL -lcurl
endif

CFLAGS += -c -std=c11 -Wall -pedantic -O2

TARGET = sia-gpu-miner

SOURCES = sia-gpu-miner.c network.c

OBJECTS = $(patsubst %.c,%.o,$(SOURCES))

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $<

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean
