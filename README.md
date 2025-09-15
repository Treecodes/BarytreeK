![Auto-test](https://github.com/pbosler/BarytreeK/actions/workflows/auto-test.yml/badge.svg)

# BarytreeK
A c++/kokkos implementation of [Barytree](https://github.com/Treecodes/BaryTree).


## Dependencies

BarytreeK is built upon the [Trilinos](https://trilinos.github.io) packages Kokkos (for performance portability), KokkosKernels (for batched dense linear algebra) and Zoltan2 (for distributed parallel domain decompostion).  Zoltan2 has required dependencies on Teuchos (common computing tasks) and Tpetra (linear algebra), amongst others.  Since BarytreeK will support the meshfree solution of PDEs, we recommend also building/linking the Compadre Trilinos package.   The current auto-test workflow is built against Trilinos version 16.1.0.

BarytreeK uses the [spdlog](https://github.com/gabime/spdlog) logging library (current auto-test uses version 1.13.0) for handling console and output of runtime information to a file.

BarytreeK uses [Catch2](https://github.com/catchorg/Catch2) testing framework (current auto-test uses version 3.5.3) for its unit tests and integration tests.
