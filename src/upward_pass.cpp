#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "structs.hpp"
#include "upward_pass_impl.hpp"

void upward_pass_2d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& charges, view_panel_2d& blfmm_panels, view_reall& proxy_source_weights, view_int& point_leaf_panel) {
	
	Kokkos::parallel_for(xcos.extent_int(0), base_pots_2d(xcos, ycos, charges, blfmm_panels, proxy_source_weights, point_leaf_panel, run_config.interp_degree));

	int ub, lb;
	for (int i = tree_info.levels-1; i > 0; i--) {
		lb = tree_info.level_start[i];
		ub = tree_info.level_start[i+1];
		Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), child_to_parent_2d(blfmm_panels, proxy_source_weights, run_config.interp_degree));
	}
}