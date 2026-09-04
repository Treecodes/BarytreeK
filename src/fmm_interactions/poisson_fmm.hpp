#ifndef H_FMM_POISSON_H
#define H_FMM_POISSON_H

#include <Kokkos_Core.hpp>

#include "structs.hpp"
#include "barytreek-config.h"

void poisson_fmm_interactions(const RunConfig& run_config, view_real& xcos, view_real& ycos, view_real& charges, view_real& soln, view_intt& panel_points_inside, 
								view_reall& proxy_source_weights, view_reall& proxy_target_weights, view_interact& pp_ints, view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, view_panel_2d& blfmm_panels);

#endif