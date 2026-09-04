#ifndef H_INTERACTION_LIST_H
#define H_INTERACTION_LIST_H

#include "structs.hpp"

void dual_tree_traversal_2d(RunConfig& run_config, view_panel_2d_host& blfmm_panels, view_interact_host& interaction_list);

void split_interactions(view_interact_host& interaction_list, view_interact_host& pp_list, view_interact_host& pc_list, view_interact_host& cp_list, view_interact_host& cc_list);

#endif