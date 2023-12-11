#!/bin/bash

rm -rf CMake*

#
#   Trilinos release 15.0.0
#

TRILINOS=$HOME/trilinos
INSTALL_DIR=$HOME/trilinos-barytreek

cmake \
-D TPL_ENABLE_MPI=ON \
-D CMAKE_C_COMPILER=mpicc \
-D CMAKE_CXX_COMPILER=mpicxx \
-D Trilinos_ENABLE_Fortran=OFF \
-D CMAKE_BUILD_TYPE=RelWithDebInfo \
-D CMAKE_INSTALL_PREFIX=$INSTALL_DIR \
-D BUILD_SHARED_LIBS=OFF \
-D Trilinos_ENABLE_ALL_PACKAGES=OFF \
-D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-D Trilinos_ENABLE_Kokkos=ON \
-D Trilinos_ENABLE_KokkosCore=ON \
-D Trilinos_ENABLE_KokkosAlgorithms=ON \
-D Trilinos_ENABLE_KokkosKernels=ON \
-D Trilinos_ENABLE_Zoltan2=ON \
-D Trilinos_ENABLE_Compadre=ON \
$TRILINOS
