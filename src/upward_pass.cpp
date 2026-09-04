#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "structs.hpp"
#include "upward_pass_impl.hpp"

void upward_pass_2d(RunConfig& run_config, view_real& xcos, view_real& ycos, view_real& charges, view_panel_2d& blfmm_panels, view_reall& proxy_source_weights, view_int& point_leaf_panel) {
	Kokkos::parallel_for(xcos.extent_int(0), base_pots(xcos, ycos, charges, blfmm_panels, proxy_source_weights, point_leaf_panel, run_config.interp_degree));

	int ub, lb;
	for (int i = run_config.blfmm_levels-1; i > 0; i--) {
		ub = run_config.blfmm_level_start[i+1];
		lb = run_config.blfmm_level_start[i];
		Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), child_to_parent(blfmm_panels, proxy_source_weights, run_config.interp_degree));
	}
}