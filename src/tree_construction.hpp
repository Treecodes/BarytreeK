#ifndef H_TREE_CONSTRUCT_H
#define H_TREE_CONSTRUCT_H

#include "barytreek-config.h"
#include "structs.hpp"

void blfmm_tree_construction(RunConfig& run_config, view_real_host& xcos, view_real_host& ycos, view_panel_2d_host& blfmm_panels, view_int_host& point_leaf_panel, view_intt_host& panel_points_inside);

#endif