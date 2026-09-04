#include "structs.hpp"
#include <queue>
#include <vector>
#include <iostream>

void dual_tree_traversal_2d(RunConfig& run_config, view_panel_2d_host& blfmm_panels_target, view_panel_2d_host& blfmm_panels_source, view_interact_host& interaction_list) {
	std::vector<interact_pair> temp_interaction_list (0);
	int interaction_count = 0;
	int pp_count = 0;
	int pc_count = 0;
	int cp_count = 0;
	int cc_count = 0;

	std::queue<int> target_panels;
	std::queue<int> source_panels;
	target_panels.push(0);
	source_panels.push(0);

	int index_target, index_source;
	bool refine_target, refine_source;
	real x1, x2, y1, y2, x_ex, y_ex, dist, separation;

	while(target_panels.size() > 0) {
		index_target = target_panels.front();
		index_source = source_panels.front();
		// std::cout << index_target << " " << index_source << std::endl;
		target_panels.pop();
		source_panels.pop();
		x1 = 0.5*(blfmm_panels_target(index_target).min_x + blfmm_panels_target(index_target).max_x);
		y1 = 0.5*(blfmm_panels_target(index_target).min_y + blfmm_panels_target(index_target).max_y);
		x2 = 0.5*(blfmm_panels_source(index_source).min_x + blfmm_panels_source(index_source).max_x);
		y2 = 0.5*(blfmm_panels_source(index_source).min_y + blfmm_panels_source(index_source).max_y);
		x_ex = x2 - x1;
		y_ex = y2 - y1;
		dist = sqrt(x_ex*x_ex + y_ex*y_ex);
		// std::cout << x_ex << " " << y_ex << " " << dist << std::endl;
		separation = 100;
		if (dist > 0) {
			separation = (blfmm_panels_target(index_target).radius + blfmm_panels_source(index_source).radius) / dist;
		}
		// std::cout << separation << std::endl;
		if (separation < run_config.fmm_theta) {
			// well separated
			interact_pair new_interact = {index_target, index_source, 0};
			if (blfmm_panels_target(index_target).point_count > run_config.fmm_cluster_thresh) {
				new_interact.interact_type += 2;
			}
			if (blfmm_panels_source(index_source).point_count > run_config.fmm_cluster_thresh) {
				new_interact.interact_type += 1;
			}
			switch (new_interact.interact_type) {
				case 0:
					pp_count += 1;
					break;
				case 1:
					pc_count += 1;
					break;
				case 2:
					cp_count += 1;
					break;
				case 3: 
					cc_count += 1;
					break;
			}
			temp_interaction_list.push_back(new_interact);
			interaction_count += 1;
			// break;
		} else {
			// not well separated
			refine_target = false;
			refine_source = false;
			if (blfmm_panels_target(index_target).is_leaf and blfmm_panels_source(index_source).is_leaf) {
				// both leaves
				interact_pair new_interact = {index_target, index_source, 0};
				temp_interaction_list.push_back(new_interact);
				interaction_count += 1;
				pp_count += 1;
			} else if (blfmm_panels_target(index_target).is_leaf) {
				// break up source panel
				refine_source = true;
			} else if (blfmm_panels_source(index_source).is_leaf) {
				refine_target = true;
			} else {
				if (blfmm_panels_target(index_target).point_count > blfmm_panels_source(index_source).point_count) {
					// target has more points, refine
					refine_target = true;
				} else {
					// source has more points, refine
					refine_source = true;
				}
			}
			if (refine_target) {
				source_panels.push(index_source);
				source_panels.push(index_source);
				source_panels.push(index_source);
				source_panels.push(index_source);
				target_panels.push(blfmm_panels_target(index_target).child1);
				target_panels.push(blfmm_panels_target(index_target).child2);
				target_panels.push(blfmm_panels_target(index_target).child3);
				target_panels.push(blfmm_panels_target(index_target).child4);
			} else if (refine_source) {
				target_panels.push(index_target);
				target_panels.push(index_target);
				target_panels.push(index_target);
				target_panels.push(index_target);
				source_panels.push(blfmm_panels_source(index_source).child1);
				source_panels.push(blfmm_panels_source(index_source).child2);
				source_panels.push(blfmm_panels_source(index_source).child3);
				source_panels.push(blfmm_panels_source(index_source).child4);
			}
		}
	}
	Kokkos::resize(interaction_list, interaction_count);
	run_config.fmm_interaction_count = interaction_count;
	for (int i = 0; i < run_config.fmm_interaction_count; i++) {
		interaction_list(i) = temp_interaction_list[i];
	}
	run_config.fmm_pp_count = pp_count;
	run_config.fmm_pc_count = pc_count;
	run_config.fmm_cp_count = cp_count;
	run_config.fmm_cc_count = cc_count;
}

void split_interactions(view_interact_host& interaction_list, view_interact_host& pp_list, view_interact_host& pc_list, view_interact_host& cp_list, view_interact_host& cc_list) {
	// splits the interactions into four separate lists
	int pp_loc = 0;
	int pc_loc = 0;
	int cp_loc = 0;
	int cc_loc = 0;
	for (int i = 0; i < interaction_list.extent_int(0); i++) {
		switch (interaction_list(i).interact_type) {
			case 0: 
				pp_list(pp_loc) = interaction_list(i);
				pp_loc += 1;
				break;
			case 1:
				pc_list(pc_loc) = interaction_list(i);
				pc_loc += 1;
				break;
			case 2:
				cp_list(cp_loc) = interaction_list(i);
				cp_loc += 1;
				break;
			case 3:
				cc_list(cc_loc) = interaction_list(i);
				cc_loc += 1;
				break;
		}
	}
}