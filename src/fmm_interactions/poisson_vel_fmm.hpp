#ifndef H_FMM_POISSON_VEL_H
#define H_FMM_POISSON_VEL_H

#include <Kokkos_Core.hpp>

#include "structs.hpp"
#include "barytreek-config.h"

void poisson_vel_fmm_interactions_3d(const RunConfig& run_config, view_real& xcos_t, view_real& ycos_t, view_real& zcos_t, view_real& xcos_s, view_real& ycos_s, view_real& zcos_s, view_real& charges, view_real& vel_x, view_real& vel_y, view_real& vel_z, view_intt& panel_points_inside_source, 
								view_intt& panel_points_inside_target, view_reall& proxy_source_weights, view_reall& proxy_target_weights_x, view_reall& proxy_target_weights_y, view_reall& proxy_target_weights_z, view_interact& pp_ints, 
								view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, view_panel_3d& blfmm_panels_source, view_panel_3d& blfmm_panels_target);

#endif