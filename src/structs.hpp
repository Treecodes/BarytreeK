#ifndef H_STRUCTS_H
#define H_STRUCTS_H

#include "barytreek-config.h"

struct RunConfig {
	int blfmm_panel_count; // number of panels in the BLFMM tree
	int blfmm_levels; // number of levels in the BLFMM tree
	int* blfmm_level_start; // start index of each level in the BLFMM tree
	int interp_degree; // interpolation degree to use for BLFMM, namelist parameter
	int interp_point_count; // number of interpolation points
	real fmm_theta = 0.7; // MAC parameter for well separated threshold, namelist parameter
	int fmm_cluster_thresh; // point count threshold to treat a panel as a cluster
	int fmm_interaction_count; // number of interactions
	int fmm_pp_count; // number of particle particle interactions
	int fmm_pc_count; // number of particle cluster interactions
	int fmm_cp_count; // number of cluster particle interactions
	int fmm_cc_count; // number of cluster cluster interactions
	real ker_eps = -1; // kernel regularization epsilon if used

	int mpi_p; // total number of MPI ranks
	int mpi_id; // MPI id
};

struct TreePanel_2d {
	int id;
	int level;
	int parent_id = -1;
	int child1 = -1;
	int child2 = -1;
	int child3 = -1;
	int child4 = -1;
	bool is_leaf = true;
	real max_x;
	real min_x;
	real max_y;
	real min_y;
	real radius;
	int point_count = 0;
};

struct TreePanel_3d {
	int id;
	int level;
	int parent_id = -1;
	int child1 = -1;
	int child2 = -1;
	int child3 = -1;
	int child4 = -1;
	int child5 = -1;
	int child6 = -1;
	int child7 = -1;
	int child8 = -1;
	bool is_leaf = true;
	real max_x;
	real min_x;
	real max_y;
	real min_y;
	real min_z;
	real max_z;
	real radius;
	int point_count = 0;
};

using view_panel_2d_host = Kokkos::View<TreePanel_2d*, Kokkos::HostSpace>;
using view_panel_2d = Kokkos::View<TreePanel_2d*>;
using view_panel_3d_host = Kokkos::View<TreePanel_3d*, Kokkos::HostSpace>;
using view_panel_3d = Kokkos::View<TreePanel_3d*>;

struct interact_pair {
	int target_panel;
	int source_panel;
	int interact_type; // 0 = pp, 1 = pc, 2 = cp, 3 = cc
};

using view_interact_host = Kokkos::View<interact_pair*, Kokkos::HostSpace>;
using view_interact = Kokkos::View<interact_pair*>;

#endif