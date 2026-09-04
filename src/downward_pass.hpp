#ifndef H_DOWNWARD_PASS_H
#define H_DOWNWARD_PASS_H

#include "barytreek-config.h"
#include "structs.hpp"

void downward_pass(RunConfig& run_config, view_real& xcos, view_real& ycos, view_real& soln, view_panel_2d& blfmm_panels, view_reall& proxy_target_weights, view_intt& panel_points_inside);

#endif