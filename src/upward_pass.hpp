#ifndef H_UPWARD_PASS_H
#define H_UPWARD_PASS_H

#include "barytreek-config.h"
#include "structs.hpp"

void upward_pass_2d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& charges, view_panel_2d& blfmm_panels, view_reall& proxy_source_weights, view_int& point_leaf_panel);

void upward_pass_3d(const RunConfig& run_config, const TreeInfo& tree_info, view_real& xcos, view_real& ycos, view_real& zcos, view_real& charges, view_panel_3d& blfmm_panels, view_reall& proxy_source_weights, view_int& point_leaf_panel);

#endif