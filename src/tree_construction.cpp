#include "structs.hpp"
#include <vector>
#include <Kokkos_Core.hpp>

void blfmm_tree_construction_2d(const RunConfig& run_config, TreeInfo& tree_info, view_real_host& xcos, view_real_host& ycos, 
								view_panel_2d_host& blfmm_panels, view_int_host& point_leaf_panel, view_intt_host& panel_points_inside) {
	std::vector<TreePanel_2d> temp_cube_panels (1);
	std::vector<std::vector<int>> temp_points_inside (1);

	int point_count = xcos.extent_int(0);

	// base panel includes all the points
	temp_cube_panels[0].id = 0;
	temp_cube_panels[0].level = 0;
	temp_cube_panels[0].point_count = point_count;
	temp_cube_panels[0].is_leaf = true;

	real min_x = std::numeric_limits<real>::max();
	real max_x = std::numeric_limits<real>::min();
	real min_y = std::numeric_limits<real>::max();
	real max_y = std::numeric_limits<real>::min();

	for (int i = 0; i < point_count; i++) {
		point_leaf_panel(i) = 0;
		temp_points_inside[0].push_back(i);
		min_x = std::min(min_x, xcos(i));
		max_x = std::max(max_x, xcos(i));
		min_y = std::min(min_y, ycos(i));
		max_y = std::max(max_y, ycos(i));
	}
	temp_cube_panels[0].min_x = min_x;
	temp_cube_panels[0].max_x = max_x;
	temp_cube_panels[0].min_y = min_y;
	temp_cube_panels[0].max_y = max_y;

	int level, points_to_assign, child_index, size, which_child;
	real x1, x2, x3, y1, y2, y3, x, y, x_ex, y_ex;
	for (int i = 0; i < temp_cube_panels.size(); i++) { // iterate through the tree
		if ((temp_cube_panels[i].point_count > run_config.fmm_cluster_thresh) and (temp_cube_panels[i].is_leaf)) {
			// refine panel
			temp_cube_panels[i].is_leaf = false;
			size = temp_cube_panels.size();
			level = temp_cube_panels[i].level;
			TreePanel_2d child1, child2, child3, child4; // 4 new child panels
			x1 = temp_cube_panels[i].min_x;
			x3 = temp_cube_panels[i].max_x;
			y1 = temp_cube_panels[i].min_y;
			y3 = temp_cube_panels[i].max_y;
			x2 = 0.5*(x1+x3);
			y2 = 0.5*(y1+y3);
			child1.min_x = x2; child1.max_x = x3; child1.min_y = y2; child1.max_y = y3; // top right
			child2.min_x = x1; child2.max_x = x2; child2.min_y = y2; child2.max_y = y3; // top left
			child3.min_x = x1; child3.max_x = x2; child3.min_y = y1; child3.max_y = y2; // bottom left
			child4.min_x = x2; child4.max_x = x3; child4.min_y = y1; child4.max_y = y2; // bottom right

			child1.parent_id = i; child2.parent_id = i; child3.parent_id = i; child4.parent_id = i;
			child1.level = level+1; child2.level = level+1; child3.level = level+1; child4.level = level+1;
			child1.id = size;
			child2.id = size+1;
			child3.id = size+2;
			child4.id = size+3;
			temp_cube_panels[i].child1 = child1.id;
			temp_cube_panels[i].child2 = child2.id;
			temp_cube_panels[i].child3 = child3.id;
			temp_cube_panels[i].child4 = child4.id;
			temp_cube_panels.push_back(child1);
			temp_cube_panels.push_back(child2);
			temp_cube_panels.push_back(child3);
			temp_cube_panels.push_back(child4);

			// assign points to new child panels
			points_to_assign = temp_cube_panels[i].point_count;
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			for (int j = 0; j < points_to_assign; j++) {
				child_index = temp_points_inside[i][j];
				x = xcos(child_index);
				y = ycos(child_index);
				if (x > x2) {
					if (y > y2) {
						which_child = 0;
					} else {
						which_child = 3;
					}
				} else {
					if (y > y2) {
						which_child = 1;
					} else {
						which_child = 2;
					}
				}

				temp_cube_panels[size+which_child].point_count += 1;
				temp_points_inside[size+which_child].push_back(child_index);
				point_leaf_panel(child_index) = size+which_child;
			}
		}
	}

	int levels = temp_cube_panels.back().level + 1;
	tree_info.level_start = (int*) malloc((levels+1) * sizeof(int));
	tree_info.levels = levels;
	tree_info.level_start[0] = 0;
	tree_info.panel_count = temp_cube_panels.size();
	int target_level = 1;

	// copy the vector of panels to the Kokkos view
	Kokkos::resize(blfmm_panels, tree_info.panel_count);
	Kokkos::resize(panel_points_inside, tree_info.panel_count, run_config.fmm_cluster_thresh);

	for (int i = 0; i < tree_info.panel_count; i++) {
		blfmm_panels(i) = temp_cube_panels[i];
		if (blfmm_panels(i).is_leaf) {
			for (int j = 0; j < blfmm_panels(i).point_count; j++) {
				panel_points_inside(i,j) = temp_points_inside[i][j];
			}
		}
		x_ex = blfmm_panels(i).max_x - blfmm_panels(i).min_x;
		y_ex = blfmm_panels(i).max_y - blfmm_panels(i).min_y;
		blfmm_panels(i).radius = 0.5*sqrt(x_ex*x_ex + y_ex*y_ex);
		if (blfmm_panels(i).level == target_level) {
			tree_info.level_start[target_level] = i;
			target_level += 1;
		}
	}
	tree_info.level_start[levels] = tree_info.panel_count;
}	

void blfmm_tree_construction_3d(const RunConfig& run_config, TreeInfo& tree_info, view_real_host& xcos, view_real_host& ycos, 
								view_real_host& zcos, view_panel_3d_host& blfmm_panels, view_int_host& point_leaf_panel, 
								view_intt_host& panel_points_inside) {
	std::vector<TreePanel_3d> temp_cube_panels (1);
	std::vector<std::vector<int>> temp_points_inside (1);

	int point_count = xcos.extent_int(0);

	// base panel includes all the points
	temp_cube_panels[0].id = 0;
	temp_cube_panels[0].level = 0;
	temp_cube_panels[0].point_count = point_count;
	temp_cube_panels[0].is_leaf = true;

	real min_x = std::numeric_limits<real>::max();
	real max_x = std::numeric_limits<real>::min();
	real min_y = std::numeric_limits<real>::max();
	real max_y = std::numeric_limits<real>::min();
	real min_z = std::numeric_limits<real>::max();
	real max_z = std::numeric_limits<real>::min();

	for (int i = 0; i < point_count; i++) {
		point_leaf_panel(i) = 0;
		temp_points_inside[0].push_back(i);
		min_x = std::min(min_x, xcos(i));
		max_x = std::max(max_x, xcos(i));
		min_y = std::min(min_y, ycos(i));
		max_y = std::max(max_y, ycos(i));
		min_z = std::min(min_z, zcos(i));
		max_z = std::max(max_z, zcos(i));
	}
	temp_cube_panels[0].min_x = min_x;
	temp_cube_panels[0].max_x = max_x;
	temp_cube_panels[0].min_y = min_y;
	temp_cube_panels[0].max_y = max_y;
	temp_cube_panels[0].min_z = min_z;
	temp_cube_panels[0].max_z = max_z;

	int level, points_to_assign, child_index, size, which_child;
	real x1, x2, x3, y1, y2, y3, z1, z2, z3, x, y, z, x_ex, y_ex, z_ex;
	for (int i = 0; i < temp_cube_panels.size(); i++) { // iterate through the tree
		if ((temp_cube_panels[i].point_count > run_config.fmm_cluster_thresh) and (temp_cube_panels[i].is_leaf)) {
			// refine panel
			temp_cube_panels[i].is_leaf = false;
			size = temp_cube_panels.size();
			level = temp_cube_panels[i].level;
			TreePanel_3d child1, child2, child3, child4, child5, child6, child7, child8; // 8 new child panels
			x1 = temp_cube_panels[i].min_x;
			x3 = temp_cube_panels[i].max_x;
			y1 = temp_cube_panels[i].min_y;
			y3 = temp_cube_panels[i].max_y;
			z1 = temp_cube_panels[i].min_z;
			z3 = temp_cube_panels[i].max_z;
			x2 = 0.5*(x1+x3);
			y2 = 0.5*(y1+y3);
			z2 = 0.5*(z1+z3);
			child1.min_x = x2; child1.max_x = x3; child1.min_y = y2; child1.max_y = y3; child1.min_z = z2; child1.max_z = z3; // top right
			child2.min_x = x1; child2.max_x = x2; child2.min_y = y2; child2.max_y = y3; child2.min_z = z2; child2.max_z = z3; // top left
			child3.min_x = x1; child3.max_x = x2; child3.min_y = y1; child3.max_y = y2; child3.min_z = z2; child3.max_z = z3; // bottom left
			child4.min_x = x2; child4.max_x = x3; child4.min_y = y1; child4.max_y = y2; child4.min_z = z2; child4.max_z = z3; // bottom right
			child5.min_x = x2; child5.max_x = x3; child5.min_y = y2; child5.max_y = y3; child5.min_z = z1; child5.max_z = z2; // top right
			child6.min_x = x1; child6.max_x = x2; child6.min_y = y2; child6.max_y = y3; child6.min_z = z1; child6.max_z = z2; // top left
			child7.min_x = x1; child7.max_x = x2; child7.min_y = y1; child7.max_y = y2; child7.min_z = z1; child7.max_z = z2; // bottom left
			child8.min_x = x2; child8.max_x = x3; child8.min_y = y1; child8.max_y = y2; child8.min_z = z1; child8.max_z = z2; // bottom right

			child1.parent_id = i; child2.parent_id = i; child3.parent_id = i; child4.parent_id = i;
			child5.parent_id = i; child6.parent_id = i; child7.parent_id = i; child8.parent_id = i;
			child1.level = level+1; child2.level = level+1; child3.level = level+1; child4.level = level+1;
			child5.level = level+1; child6.level = level+1; child7.level = level+1; child8.level = level+1;
			child1.id = size;
			child2.id = size+1;
			child3.id = size+2;
			child4.id = size+3;
			child5.id = size+4;
			child6.id = size+5;
			child7.id = size+6;
			child8.id = size+7;
			temp_cube_panels[i].child1 = child1.id;
			temp_cube_panels[i].child2 = child2.id;
			temp_cube_panels[i].child3 = child3.id;
			temp_cube_panels[i].child4 = child4.id;
			temp_cube_panels[i].child5 = child5.id;
			temp_cube_panels[i].child6 = child6.id;
			temp_cube_panels[i].child7 = child7.id;
			temp_cube_panels[i].child8 = child8.id;
			temp_cube_panels.push_back(child1);
			temp_cube_panels.push_back(child2);
			temp_cube_panels.push_back(child3);
			temp_cube_panels.push_back(child4);
			temp_cube_panels.push_back(child5);
			temp_cube_panels.push_back(child6);
			temp_cube_panels.push_back(child7);
			temp_cube_panels.push_back(child8);

			// assign points to new child panels
			points_to_assign = temp_cube_panels[i].point_count;
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			temp_points_inside.push_back(std::vector<int> (0));
			for (int j = 0; j < points_to_assign; j++) {
				child_index = temp_points_inside[i][j];
				x = xcos(child_index);
				y = ycos(child_index);
				z = zcos(child_index);
				if (z > z2) {
					if (x > x2) {
						if (y > y2) {
							which_child = 0;
						} else {
							which_child = 3;
						}
					} else {
						if (y > y2) {
							which_child = 1;
						} else {
							which_child = 2;
						}
					}
				} else {
					if (x > x2) {
						if (y > y2) {
							which_child = 4;
						} else {
							which_child = 7;
						}
					} else {
						if (y > y2) {
							which_child = 5;
						} else {
							which_child = 6;
						}
					}
				}
				

				temp_cube_panels[size+which_child].point_count += 1;
				temp_points_inside[size+which_child].push_back(child_index);
				point_leaf_panel(child_index) = size+which_child;
			}
		}
	}

	int levels = temp_cube_panels.back().level + 1;
	tree_info.level_start = (int*) malloc((levels+1) * sizeof(int));
	tree_info.levels = levels;
	tree_info.level_start[0] = 0;
	tree_info.panel_count = temp_cube_panels.size();
	int target_level = 1;

	// copy the vector of panels to the Kokkos view
	Kokkos::resize(blfmm_panels, tree_info.panel_count);
	Kokkos::resize(panel_points_inside, tree_info.panel_count, run_config.fmm_cluster_thresh);

	for (int i = 0; i < tree_info.panel_count; i++) {
		blfmm_panels(i) = temp_cube_panels[i];
		if (blfmm_panels(i).is_leaf) {
			for (int j = 0; j < blfmm_panels(i).point_count; j++) {
				panel_points_inside(i,j) = temp_points_inside[i][j];
			}
		}
		x_ex = blfmm_panels(i).max_x - blfmm_panels(i).min_x;
		y_ex = blfmm_panels(i).max_y - blfmm_panels(i).min_y;
		z_ex = blfmm_panels(i).max_z - blfmm_panels(i).min_z;
		blfmm_panels(i).radius = 0.5*sqrt(x_ex*x_ex + y_ex*y_ex + z_ex*z_ex);
		if (blfmm_panels(i).level == target_level) {
			tree_info.level_start[target_level] = i;
			target_level += 1;
		}
	}
	tree_info.level_start[levels] = tree_info.panel_count;
}	