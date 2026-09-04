#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "barytreek-config.h"
#include "structs.hpp"

void read_run_config(const std::string file_name, RunConfig& run_config) {
	std::ifstream config_file(file_name);
	if (config_file.fail()) {
	    std::cout << "namelist file at " << file_name << std::endl;
	    throw std::runtime_error("namelist not found");
	}
	std::string line, word1, word2;

	while (true) {
		getline(config_file, line);
		std::stringstream str1(line);
		getline(str1, word1, '=');
		getline(str1, word2);

		if (word1 == "interp_degree") {
			run_config.interp_degree = stoi(word2);
			if (run_config.interp_degree > max_degree) {
				run_config.interp_degree = max_degree;
			}
			run_config.interp_point_count = (run_config.interp_degree+1) * (run_config.interp_degree+1);
		} else if (word1 == "theta") {
			run_config.fmm_theta = stod(word2);
		} else if (word1 == "kernel_eps") {
			run_config.ker_eps = stod(word2);
		} else {
			run_config.fmm_cluster_thresh = 4 * run_config.interp_point_count;
			return;
		}
	}
}