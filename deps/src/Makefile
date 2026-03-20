# Makefile for FastHAL C library (Phase 1: Core basis operations)
# Targets: all, test, clean, test-verbose

CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c99 -fPIC -g
LDFLAGS = -lm -fopenmp

# Optional BLAS/LAPACK support (detected at runtime or compile-time)
# Uncomment to enable:
# CFLAGS += -DUSE_BLAS
# LDFLAGS += -lblas -llapack

# CUnit framework (for testing)
CUNIT_CFLAGS = $(shell pkg-config --cflags cunit 2>/dev/null)
CUNIT_LIBS = $(shell pkg-config --libs cunit 2>/dev/null)
CUNIT_AVAILABLE = $(shell pkg-config --exists cunit && echo yes || echo no)

SRCDIR = .
TESTDIR = test
OBJDIR = .objbuild
TESTOBJ = $(OBJDIR)/test

# Source files for library
LIB_SOURCES = common.h memory_pool.c rank_and_path.c nested_matrix.c basis_matrix.c coordinate_descent.c fit_model.c c_julia_wrapper.c
LIB_OBJECTS = $(patsubst %.c,$(OBJDIR)/%.o,$(filter %.c,$(LIB_SOURCES)))

# Library targets
LIB_STATIC = libfasthal.a
LIB_SHARED = libfasthal.so

# All targets
all: $(LIB_STATIC) $(LIB_SHARED)

# Object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR) $(TESTOBJ)

# Static library
$(LIB_STATIC): $(LIB_OBJECTS)
	ar rcs $@ $^

# Shared library
$(LIB_SHARED): $(LIB_OBJECTS)
	$(CC) -shared -fPIC -o $@ $^ $(LDFLAGS)

# Testing setup
ifeq ($(CUNIT_AVAILABLE),yes)

# Test sources
TEST_SOURCES = $(TESTDIR)/test_nested_matrix.c $(TESTDIR)/test_rank_and_path.c $(TESTDIR)/test_basis_matrix.c $(TESTDIR)/test_coordinate_descent.c $(TESTDIR)/run_tests.c
TEST_OBJECTS = $(patsubst $(TESTDIR)/%.c,$(TESTOBJ)/%.o,$(TEST_SOURCES))
TEST_FIXTURE_OBJECTS = $(TESTOBJ)/test_fixtures.o
TEST_RUNNER = run_tests

# Test object files
$(TESTOBJ)/%.o: $(TESTDIR)/%.c | $(TESTOBJ)
	$(CC) $(CFLAGS) $(CUNIT_CFLAGS) -c $< -o $@

$(TESTOBJ)/test_fixtures.o: test_fixtures.c | $(TESTOBJ)
	$(CC) $(CFLAGS) $(CUNIT_CFLAGS) -c $< -o $@

# Test executable
$(TEST_RUNNER): $(LIB_STATIC) $(TEST_OBJECTS) $(TEST_FIXTURE_OBJECTS) | $(TESTOBJ)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(TEST_FIXTURE_OBJECTS) $(LIB_STATIC) $(CUNIT_LIBS) $(LDFLAGS)

test: $(TEST_RUNNER)
	./$(TEST_RUNNER)

test-verbose: $(TEST_RUNNER)
	./$(TEST_RUNNER) -v

test-memory: $(TEST_RUNNER)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_RUNNER)

else

test:
	@echo "CUnit not found. Install CUnit and pkg-config to run tests."
	@echo "  Ubuntu/Debian: sudo apt-get install libcunit1-dev pkg-config"
	@echo "  macOS: brew install cunit pkg-config"

test-verbose: test
test-memory: test

endif

# Clean
clean:
	rm -rf $(OBJDIR) $(LIB_STATIC) $(LIB_SHARED) $(TEST_RUNNER)
	rm -f *.o *.so *.a

.PHONY: all test test-verbose test-memory clean

