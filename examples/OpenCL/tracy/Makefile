ifndef CPPC
	CPPC=clang++
endif

CCFLAGS=

INC = -I $(CPP_COMMON)

LIBS = -lOpenCL -lrt

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

TRACY_DIR = ../../../../tracy/public

# For Tracy
CCFLAGS += -g -fno-omit-frame-pointer -rdynamic

OpenCLVectorAdd: OpenCLVectorAdd.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -I $(TRACY_DIR) -o $@

clean:
	rm -f OpenCLVectorAdd
