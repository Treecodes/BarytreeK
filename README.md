# BarytreeK
A c++/kokkos implementation of [Barytree](https://github.com/Treecodes/BaryTree).


## Dependencies

BarytreeK is built upon the [Trilinos](https://trilinos.github.io) version 15.0.0 packages Kokkos (for performance portability) and Zoltan2 (for distributed parallel domain decompostion).  Zoltan2 has required dependencies on Teuchos (common computing tasks) and Tpetra (linear algebra).

BarytreeK uses the [spdlog](https://github.com/gabime/spdlog) logging library version 1.12.0 for handling console and output of runtime information to a file.

BarytreeK uses the [GoogleTest](https://github.com/google/googletest) testing framework, version 1.14.0.