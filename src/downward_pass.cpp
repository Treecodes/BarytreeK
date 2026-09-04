#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "structs.hpp"
#include "downward_pass_impl.hpp"

void downward_pass_2d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& soln, view_panel_2d& blfmm_panels, view_reall& proxy_target_weights, view_intt& panel_points_inside) {
	int lb, ub;
	for (int i = 0; i < tree_info.levels-1; i++) {
		lb = tree_info.level_start[i];
		ub = tree_info.level_start[i+1];
		Kokkos::parallel_for(Kokkos::MDRangePolicy({lb, 0},{ub, 4}), parent_to_child_2d(blfmm_panels, proxy_target_weights, run_config.interp_degree));
	}
	lb = tree_info.level_start[tree_info.levels-1];
	ub = tree_info.level_start[tree_info.levels];
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), leaf_to_point_2d(xcos, ycos, soln, proxy_target_weights, blfmm_panels, panel_points_inside, run_config.interp_degree));
}