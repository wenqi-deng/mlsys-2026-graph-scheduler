CXX = g++
CXXFLAGS = -std=c++20 -Wall -MMD -MP -O2
PYTHON ?= python3

SRC_DIR = source
BUILD_DIR = build
TARGET = $(BUILD_DIR)/mlsys

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

DATA_DIR = benchmarks
OUTPUT_DIR = output
VERIFY_SCRIPT = tools/verify_solution.py

DATA_FILES = $(wildcard $(DATA_DIR)/*.json)
OUTPUT_FILES = $(DATA_FILES:$(DATA_DIR)/%.json=$(OUTPUT_DIR)/%.json)

.PHONY: all run verify clean tar

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(DEPS)


run: $(TARGET) $(OUTPUT_FILES)

verify: run
	@set -e; \
	for in_file in $(DATA_FILES); do \
		out_file=$(OUTPUT_DIR)/$$(basename $$in_file); \
		echo "Verifying $$in_file -> $$out_file"; \
		$(PYTHON) $(VERIFY_SCRIPT) $$in_file $$out_file; \
	done

$(OUTPUT_DIR)/%.json: $(DATA_DIR)/%.json $(TARGET) | $(OUTPUT_DIR)
	$(TARGET) $< $@

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR) *.tar

tar:
	tar -cvf source.tar source/ Makefile doc/

