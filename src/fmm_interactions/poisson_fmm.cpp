#include <Kokkos_Core.hpp>
#include <iostream>

#include "structs.hpp"
#include "barytreek-config.h"
#include "poisson_fmm_impl.hpp"

void poisson_fmm_interactions_2d(const RunConfig& run_config, view_real& xcos_t, view_real& ycos_t, view_real& xcos_s, view_real& ycos_s, view_real& charges, view_real& soln, view_intt& panel_points_inside_source, view_intt& panel_points_inside_target, 
								view_reall& proxy_source_weights, view_reall& proxy_target_weights, view_interact& pp_ints, view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, 
								view_panel_2d& blfmm_panels_source, view_panel_2d& blfmm_panels_target) {
	Kokkos::parallel_for(run_config.fmm_pp_count, poisson_pp_interaction_2d(xcos_t, ycos_t, xcos_s, ycos_s, charges, soln, panel_points_inside_target, panel_points_inside_source, pp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps));
	Kokkos::parallel_for(run_config.fmm_pc_count, poisson_pc_interaction_2d(xcos_t, ycos_t, soln, panel_points_inside_target, proxy_source_weights, pc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cp_count, poisson_cp_interaction_2d(xcos_s, ycos_s, charges, panel_points_inside_source, proxy_target_weights, cp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cc_count, poisson_cc_interaction_2d(proxy_target_weights, proxy_source_weights, cc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::fence();
}

void poisson_fmm_interactions_3d(const RunConfig& run_config, view_real& xcos_t, view_real& ycos_t, view_real& zcos_t, view_real& xcos_s, view_real& ycos_s, view_real& zcos_s, view_real& charges, view_real& soln, view_intt& panel_points_inside_source, view_intt& panel_points_inside_target, 
								view_reall& proxy_source_weights, view_reall& proxy_target_weights, view_interact& pp_ints, view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, 
								view_panel_3d& blfmm_panels_source, view_panel_3d& blfmm_panels_target) {
	Kokkos::parallel_for(run_config.fmm_pp_count, poisson_pp_interaction_3d(xcos_t, ycos_t, zcos_t, xcos_s, ycos_s, zcos_s, charges, soln, panel_points_inside_target, panel_points_inside_source, pp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps));
	Kokkos::parallel_for(run_config.fmm_pc_count, poisson_pc_interaction_3d(xcos_t, ycos_t, zcos_t, soln, panel_points_inside_target, proxy_source_weights, pc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cp_count, poisson_cp_interaction_3d(xcos_s, ycos_s, zcos_s, charges, panel_points_inside_source, proxy_target_weights, cp_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::parallel_for(run_config.fmm_cc_count, poisson_cc_interaction_3d(proxy_target_weights, proxy_source_weights, cc_ints, blfmm_panels_target, blfmm_panels_source, run_config.ker_eps, run_config.interp_degree));
	Kokkos::fence();
}