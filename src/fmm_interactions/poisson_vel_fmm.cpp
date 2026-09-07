#include <Kokkos_Core.hpp>
#include <iostream>

#include "structs.hpp"
#include "barytreek-config.h"
#include "poisson_vel_fmm_impl.hpp"

void poisson_vel_fmm_interactions_3d(const RunConfig& run_config, view_real& xcos_t, view_real& ycos_t, view_real& zcos_t, view_real& xcos_s, view_real& ycos_s, view_real& zcos_s, view_real& charges, view_real& vel_x, view_real& vel_y, view_real& vel_z, view_intt& panel_points_inside_source, 
								view_intt& panel_points_inside_target, view_reall& proxy_source_weights, view_reall& proxy_target_weights_x, view_reall& proxy_target_weights_y, view_reall& proxy_target_weights_z, view_interact& pp_ints, 
								view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, view_panel_3d& blfmm_panels_source, view_panel_3d& blfmm_panels_target) {
	// first compute list of interactions to compute
	Kokkos::parallel_for(run_config.fmm_pp_count, poisson_vel_pp_interaction_3d(xcos_t, ycos_t, zcos_t, xcos_s, ycos_s, zcos_s, charges, vel_x, vel_y, vel_z, panel_points_inside_target, panel_points_inside_source, pp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps));
	Kokkos::parallel_for(run_config.fmm_pc_count, poisson_vel_pc_interaction_3d(xcos_t, ycos_t, zcos_t, vel_x, vel_y, vel_z, panel_points_inside_target, proxy_source_weights, pc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cp_count, poisson_vel_cp_interaction_3d(xcos_s, ycos_s, zcos_s, charges, panel_points_inside_source, proxy_target_weights_x, proxy_target_weights_y, proxy_target_weights_z, cp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cc_count, poisson_vel_cc_interaction_3d(proxy_target_weights_x, proxy_target_weights_y, proxy_target_weights_z, proxy_source_weights, cc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::fence();
}