cmake_minimum_required(VERSION 3.24)

project(BarytreeK)
# We use the MPI-C interface, so we have to enable the C language
enable_language(C CXX)
set(BTK_VERSION_MAJOR 0)
set(BTK_VERSION_MINOR 1)

set(CMAKE_CXX_STANDARD 17)

# default to using <PACKAGE>_ROOT environment variables for finding third-party libraries
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0144 NEW)
include(GNUInstallDirs)

find_package(MPI REQUIRED COMPONENTS C)
find_package(Kokkos REQUIRED CONFIG)

#message(STATUS "Looking for Trilinos at ${Trilinos_DIR}, $ENV{Trilinos_ROOT}")
#find_package(Trilinos REQUIRED HINTS ${Trilinos_DIR} ${Trilinos_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/Trilinos #$ENV{Trilinos_ROOT}/${CMAKE_INSTALL_LIBDIR}/cmake/Trilinos)
# Echo trilinos build info just for fun
#MESSAGE("\nFound Trilinos!  Here are the details: ")
#MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
#MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
#MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
#MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
#MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
#MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
#MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
#MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
#MESSAGE("End of Trilinos details\n")

# have cmake treat trilinos headers as system headers (and ignore warnings)
#option(BTK_SUPRESS_TRILINOS_WARNINGS "Treat Trilinos headers as system headers" ON)

# pull Kokkos configuration from Trilinos
if (Kokkos_ENABLE_OPENMP)
    message(STATUS "Trilinos/Kokkos built with OpenMP support; BarytreeK will also")
endif()
if (Kokkos_ENABLE_THREADS)
    message(STATUS "Trilinos/Kokkos built with PThreads support; BarytreeK will also")
endif()
if (Kokkos_ENABLE_CUDA)
    message(STATUS "Trilinos/Kokkos built with Cuda support; BarytreeK will also")
endif()

#SET(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
#SET(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")

#find_package(spdlog REQUIRED HINTS ${Spdlog_DIR})

#find_package(GTest REQUIRED HINTS ${GTest_DIR})

add_subdirectory(src)
add_subdirectory(examples)

configure_file(barytreek-config.h.in barytreek-config.h)
