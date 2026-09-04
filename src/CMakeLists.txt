add_library(barytreek
    downward_pass.cpp
    fmm_interactions/poisson_fmm.cpp
    interaction_list.cpp
    read_namelist.cpp
    tree_construction.cpp
    upward_pass.cpp)

target_link_libraries(barytreek PUBLIC MPI::MPI_C Kokkos::kokkos)

target_include_directories(barytreek PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>)

set_property(TARGET barytreek PROPERTY POSITION_INDEPENDENT_CODE ON)

install(FILES
    # all header/include files go here
    ${CMAKE_BINARY_DIR}/barytreek-config.h
    bli_impl.hpp
    direct_sum_impl.hpp
    downward_pass.hpp
    downward_pass_impl.hpp
    fmm_interactions/poisson_fmm.hpp
    fmm_interactions/poisson_fmm_impl.hpp
    general_utils_impl.hpp
    interaction_list.hpp
    read_namelist.hpp
    structs.hpp
    tree_construction.hpp
    upward_pass.hpp
    upward_pass_impl.hpp
    DESTINATION ${PROJECT_SOURCE_DIR}/include)