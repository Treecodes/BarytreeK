add_executable(example_poisson_2d example_poisson_2d.cpp)
target_link_libraries(example_poisson_2d PUBLIC barytreek MPI::MPI_C Kokkos::kokkos)
target_include_directories(example_poisson_2d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

add_executable(example_poisson_2d_split example_poisson_2d_split.cpp)
target_link_libraries(example_poisson_2d_split PUBLIC barytreek MPI::MPI_C Kokkos::kokkos)
target_include_directories(example_poisson_2d_split PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)