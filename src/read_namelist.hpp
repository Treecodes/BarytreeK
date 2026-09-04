#ifndef H_READ_NAMELIST_H
#define H_READ_NAMELIST_H

#include <string>
#include "structs.hpp"

void read_run_config(const std::string file_name, RunConfig& run_config);

#endif