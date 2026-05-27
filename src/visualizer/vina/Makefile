# ============================================================
#  AutoDock-Vina (CPU) – release Makefile
#  Adapted for c1105a-s9 cluster, mirroring GPU-Vina path style
# ============================================================

# --- Cluster module paths (match your GPU-Vina Makefile) ---
BOOST_ROOT      = /apps/boost/1.77.0
BOOST_INCLUDE   = $(BOOST_ROOT)/include
BOOST_LIB_PATH  = $(BOOST_ROOT)/lib

# --- Compiler ---
GPP             = g++

# --- Compiler options ---
# Removed -static (incompatible with dynamically-linked cluster Boost)
# Added -std=c++11 (required for modern Boost) and -DBOOST_TIMER_ENABLE_DEPRECATED
C_PLATFORM      = -pthread
C_OPTIONS       = -O3 -DNDEBUG -std=c++11 -DBOOST_TIMER_ENABLE_DEPRECATED

# --- These two are used by makefile_common's linker flags ---
# Leave BOOST_VERSION blank so makefile_common doesn't append a version suffix
BOOST_VERSION   =
BOOST_LIB_VERSION =

# --- Pass library path so the linker finds /apps/boost/1.77.0/lib ---
# makefile_common uses -L$(BASE)/lib internally; we override BASE here
BASE            = $(BOOST_ROOT)

include ../../makefile_common