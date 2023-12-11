#!/bin/bash

#
#   spdlog v1.12.0
#

rm -rf CMake*

SPDLOG_SRC=$HOME/spdlog

cmake \
-D CMAKE_CXX_COMPILER=g++ \
-D CMAKE_INSTALL_PREFIX=$HOME/spdlog-barytreek \
-D SPDLOG_BUILD_EXAMPLES=OFF \
-D SPDLOG_BUILD_TESTS=OFF \
$SPDLOG_SRC
