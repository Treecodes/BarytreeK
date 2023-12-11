#!/bin/bash

rm -rf CMake*

BTK_SRC=$HOME/barytreek

Trilinos_DIR=$HOME/trilinos-barytreek
Spdlog_DIR=$HOME/spdlog-barytreek
GTest_DIR=$HOME/gtest

cmake \
-D CMAKE_BUILD_TYPE=DEBUG \
-D CMAKE_CXX_COMPILER=mpicxx \
-D CMAKE_C_COMPILER=mpicc \
-D CMAKE_INSTALL_PREFIX=./install \
-D Trilinos_DIR=$Trilinos_DIR \
-D Spdlog_DIR=$Spdlog_DIR \
-D GTest_DIR=$GTest_DIR \
$BTK_SRC