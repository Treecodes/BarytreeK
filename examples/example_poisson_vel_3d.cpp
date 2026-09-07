#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <numbers>
#include <random>

#include <Kokkos_Core.hpp>

#include "barytreek-config.h"

#include "downward_pass.hpp"
#include "fmm_interactions/poisson_vel_fmm.hpp"
#include "general_utils_impl.hpp"
#include "interaction_list.hpp"
#include "read_namelist.hpp"
#include "structs.hpp"
#include "tree_construction.hpp"
#include "upward_pass.hpp"
#include "direct_sum_impl.hpp"

// example of computing the gradient of the solution Poisson equation in 3d on 1 MPI rank
// eg gravitational acceleration
// specify the number of Kokkos threads at runtime
// for example, ./example_poisson_vel_3d --kokkos-num-threads=8

int main(int argc, char* argv[]) {
	RunConfig run_config;
	run_config.dim = 3;
	read_run_config(std::string(NAMELIST_DIR) + std::string("namelist.txt"), run_config);
	run_config.mpi_id = 0;
	run_config.mpi_p = 1;

	int point_count = 20000;

	// physical domain
	real min_x = -1.0;
	real max_x = 1.0;
	real min_y = -1.0;
	real max_y = 1.0;
	real min_z = -1.0;
	real max_z = 1.0;

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<> real_rng_x(min_x, max_x);
	std::uniform_real_distribution<> real_rng_y(min_y, max_y);
	std::uniform_real_distribution<> real_rng_z(min_z, max_z);

	std::chrono::steady_clock::time_point begin, end;

	Kokkos::initialize(argc, argv); {
		std::cout << "Kokkos num threads: " << Kokkos::num_threads() << ", Kokkos num devices: " << Kokkos::num_devices() << std::endl;

		view_real_host x_co ("x coordinates", point_count);
		view_real_host y_co ("y coordinates", point_count);
		view_real_host z_co ("z coordinates", point_count);
		view_real_host charges ("point charges", point_count);
		view_real_host vel_x ("x velocity", point_count);
		view_real_host vel_y ("y velocity", point_count);
		view_real_host vel_z ("z velocity", point_count);
		view_real_host vel_x_ds ("direct sum x velocity", point_count);
		view_real_host vel_y_ds ("direct sum y velocity", point_count);
		view_real_host vel_z_ds ("direct sum z velocity", point_count);

		// generate the points randomly
		for (int i = 0; i < point_count; i++) {
			x_co(i) = real_rng_x(rng);
			y_co(i) = real_rng_y(rng);
			z_co(i) = real_rng_z(rng);
			charges(i) = 1.0 / (1.0 + x_co(i)*x_co(i) + y_co(i)*y_co(i) + z_co(i)*z_co(i)); // whatever, made up
		}

		begin = std::chrono::steady_clock::now();

		TreeInfo tree_info;
		view_panel_3d_host blfmm_panels ("blfmm tree panels", 1); // tree structure
		view_int_host point_leaf_panel ("point leaf panel indices", point_count); // index of the leaf panel containing this point
		view_intt_host panel_points_inside ("leaf panels contained points", 1, 1); // for leaf panels, contains the indices of the contained points

		blfmm_tree_construction_3d(run_config, tree_info, x_co, y_co, z_co, blfmm_panels, point_leaf_panel, panel_points_inside);

		end = std::chrono::steady_clock::now();
		std::cout << "tree construction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		std::cout << "blfmm panels: " << tree_info.panel_count << std::endl;
		std::cout << "blfmm levels: " << tree_info.levels << std::endl;
		begin = std::chrono::steady_clock::now();

		view_interact_host interaction_list ("blfmm interactions", 1); // list of interactions

		dual_tree_traversal_3d(run_config, blfmm_panels, blfmm_panels, interaction_list);

		view_interact_host pp_interactions ("pp interactions", run_config.fmm_pp_count);
		view_interact_host pc_interactions ("pc interactions", run_config.fmm_pc_count);
		view_interact_host cp_interactions ("cp interactions", run_config.fmm_cp_count);
		view_interact_host cc_interactions ("cc interactions", run_config.fmm_cc_count);

		split_interactions(interaction_list, pp_interactions, pc_interactions, cp_interactions, cc_interactions);

		end = std::chrono::steady_clock::now();
		std::cout << "dual tree traversal time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		std::cout << "interaction count: " << run_config.fmm_interaction_count << std::endl;
		std::cout << "pp interaction count: " << run_config.fmm_pp_count << std::endl;
		std::cout << "pc interaction count: " << run_config.fmm_pc_count << std::endl;
		std::cout << "cp interaction count: " << run_config.fmm_cp_count << std::endl;
		std::cout << "cc interaction count: " << run_config.fmm_cc_count << std::endl;
		begin = std::chrono::steady_clock::now();

		view_real d_x_co ("device x coordinates", point_count);
		view_real d_y_co ("device y coordinates", point_count);
		view_real d_z_co ("device z coordinates", point_count);
		view_real d_charges ("device charges", point_count);
		view_real d_vel_x ("device x velocity", point_count);
		view_real d_vel_y ("device y velocity", point_count);
		view_real d_vel_z ("device z velocity", point_count);
		view_interact d_pp_ints ("device pp interactions", run_config.fmm_pp_count);
		view_interact d_pc_ints ("device pc interactions", run_config.fmm_pc_count);
		view_interact d_cp_ints ("device cp interactions", run_config.fmm_cp_count);
		view_interact d_cc_ints ("device cc interactions", run_config.fmm_cc_count);
		view_panel_3d d_blfmm_panels ("device blfmm panels", tree_info.panel_count);
		view_int d_point_leaf_panel ("device point leaf panel indices", point_count);
		view_intt d_panel_points_inside ("device leaf panels contained points", tree_info.panel_count, run_config.fmm_cluster_thresh);

		Kokkos::deep_copy(d_x_co, x_co);
		Kokkos::deep_copy(d_y_co, y_co);
		Kokkos::deep_copy(d_z_co, z_co);
		Kokkos::deep_copy(d_charges, charges);
		Kokkos::deep_copy(d_blfmm_panels, blfmm_panels);
		Kokkos::deep_copy(d_point_leaf_panel, point_leaf_panel);
		Kokkos::deep_copy(d_panel_points_inside, panel_points_inside);
		Kokkos::deep_copy(d_pp_ints, pp_interactions);
		Kokkos::deep_copy(d_pc_ints, pc_interactions);
		Kokkos::deep_copy(d_cp_ints, cp_interactions);
		Kokkos::deep_copy(d_cc_ints, cc_interactions);

		view_reall proxy_source_weights ("proxy source weights", tree_info.panel_count, run_config.interp_point_count);
		view_reall proxy_target_weights_x ("proxy target weights x vel", tree_info.panel_count, run_config.interp_point_count);
		view_reall proxy_target_weights_y ("proxy target weights y vel", tree_info.panel_count, run_config.interp_point_count);
		view_reall proxy_target_weights_z ("proxy target weights z vel", tree_info.panel_count, run_config.interp_point_count);

		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_source_weights, 0));
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_target_weights_x, 0));
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_target_weights_y, 0));
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_target_weights_z, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_x, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_y, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_z, 0));

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "host to device communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		upward_pass_3d(run_config, tree_info, d_x_co, d_y_co, d_z_co, d_charges, d_blfmm_panels, proxy_source_weights, d_point_leaf_panel);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "upward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		poisson_vel_fmm_interactions_3d(run_config, d_x_co, d_y_co, d_z_co, d_x_co, d_y_co, d_z_co, d_charges, d_vel_x, d_vel_y, d_vel_z, d_panel_points_inside, d_panel_points_inside, proxy_source_weights, proxy_target_weights_x, proxy_target_weights_y, proxy_target_weights_z, d_pp_ints, d_pc_ints, d_cp_ints, d_cc_ints, d_blfmm_panels, d_blfmm_panels);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "interaction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		// downward_pass_3d(run_config, tree_info, d_x_co, d_y_co, d_z_co, d_vel_x, d_blfmm_panels, proxy_target_weights_x, d_panel_points_inside);
		// downward_pass_3d(run_config, tree_info, d_x_co, d_y_co, d_z_co, d_vel_y, d_blfmm_panels, proxy_target_weights_y, d_panel_points_inside);
		// downward_pass_3d(run_config, tree_info, d_x_co, d_y_co, d_z_co, d_vel_z, d_blfmm_panels, proxy_target_weights_z, d_panel_points_inside);
		downward_pass_3d_3(run_config, tree_info, d_x_co, d_y_co, d_z_co, d_vel_x, d_vel_y, d_vel_z, d_blfmm_panels, proxy_target_weights_x, proxy_target_weights_y, proxy_target_weights_z, d_panel_points_inside);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "downward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		view_real d_vel_x_ds ("device direct sum vel x", point_count);
		view_real d_vel_y_ds ("device direct sum vel y", point_count);
		view_real d_vel_z_ds ("device direct sum vel z", point_count);
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_x_ds, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_y_ds, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_vel_z_ds, 0));

		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {point_count, point_count}), poisson_vel_3d_dir_sum(d_x_co, d_y_co, d_z_co, d_x_co, d_y_co, d_z_co, d_charges, d_vel_x_ds, d_vel_y_ds, d_vel_z_ds, run_config.ker_eps));

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "direct sum time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		Kokkos::deep_copy(vel_x, d_vel_x);
		Kokkos::deep_copy(vel_y, d_vel_y);
		Kokkos::deep_copy(vel_z, d_vel_z);
		Kokkos::deep_copy(vel_x_ds, d_vel_x_ds);
		Kokkos::deep_copy(vel_y_ds, d_vel_y_ds);
		Kokkos::deep_copy(vel_z_ds, d_vel_z_ds);

		real e1 = 0;
		real e2 = 0;

		for (int i = 0; i < point_count; i++) {
			e1 += (vel_x(i) - vel_x_ds(i)) * (vel_x(i) - vel_x_ds(i));
			e2 += vel_x_ds(i) * vel_x_ds(i);
			e1 += (vel_y(i) - vel_y_ds(i)) * (vel_y(i) - vel_y_ds(i));
			e2 += vel_y_ds(i) * vel_y_ds(i);
			e1 += (vel_y(i) - vel_y_ds(i)) * (vel_y(i) - vel_y_ds(i));
			e2 += vel_y_ds(i) * vel_y_ds(i);
		}

		std::cout << "relative l2 error is " << sqrt(e1/e2) << std::endl;
	}
	Kokkos::finalize();
	return 0;
}