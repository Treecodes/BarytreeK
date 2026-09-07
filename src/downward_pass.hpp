#ifndef H_DOWNWARD_PASS_H
#define H_DOWNWARD_PASS_H

#include "barytreek-config.h"
#include "structs.hpp"

void downward_pass_2d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& soln, view_panel_2d& blfmm_panels, view_reall& proxy_target_weights, view_intt& panel_points_inside);

void downward_pass_3d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& zcos, view_real& soln, view_panel_3d& blfmm_panels, view_reall& proxy_target_weights, view_intt& panel_points_inside);

void downward_pass_3d_3(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& zcos, view_real& vel_x, view_real& vel_y, view_real& vel_z, view_panel_3d& blfmm_panels, view_reall& proxy_target_weights_x, view_reall& proxy_target_weights_y, view_reall& proxy_target_weights_z, view_intt& panel_points_inside);

#endif