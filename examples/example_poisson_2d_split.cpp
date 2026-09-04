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
#include "fmm_interactions/poisson_fmm.hpp"
#include "general_utils_impl.hpp"
#include "interaction_list.hpp"
#include "read_namelist.hpp"
#include "structs.hpp"
#include "tree_construction.hpp"
#include "upward_pass.hpp"
#include "direct_sum_impl.hpp"

// example of solving the Poisson equation in 2d on 1 MPI rank
// with separate sources and targets
// specify the number of Kokkos threads at runtime
// for example, ./example_poisson_2d --kokkos-num-threads=8

int main(int argc, char* argv[]) {
	RunConfig run_config;
	read_run_config(std::string(NAMELIST_DIR) + std::string("namelist.txt"), run_config);
	run_config.mpi_id = 0;
	run_config.mpi_p = 1;

	int point_count = 10000;

	// physical domain
	real min_x = -1.0;
	real max_x = 1.0;
	real min_y = -1.0;
	real max_y = 1.0;

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<> real_rng_x(min_x, max_x);
	std::uniform_real_distribution<> real_rng_y(min_y, max_y);

	std::chrono::steady_clock::time_point begin, end;

	Kokkos::initialize(argc, argv); {
		std::cout << "kokkos num threads: " << Kokkos::num_threads() << ", Kokkos num devices: " << Kokkos::num_devices() << std::endl;

		view_real_host x_co_t ("target x coordinates", point_count);
		view_real_host y_co_t ("target y coordinates", point_count);
		view_real_host x_co_s ("source x coordinates", point_count);
		view_real_host y_co_s ("source y coordinates", point_count);
		view_real_host charges ("point charges", point_count);
		view_real_host sols ("solution", point_count);
		view_real_host sols_ds ("direct sum solution", point_count);

		// generate the points randomly
		for (int i = 0; i < point_count; i++) {
			x_co_t(i) = real_rng_x(rng);
			y_co_t(i) = real_rng_y(rng);
			x_co_s(i) = real_rng_x(rng);
			y_co_s(i) = real_rng_y(rng);
			charges(i) = 1.0 / (1.0 + x_co_s(i)*x_co_s(i) + y_co_s(i)*y_co_s(i)); // whatever, made up
		}

		begin = std::chrono::steady_clock::now();

		TreeInfo tree_info_target, tree_info_source;
		view_panel_2d_host blfmm_panels_target ("target blfmm tree panels", 1); // tree structure for target points
		view_panel_2d_host blfmm_panels_source ("source blfmm tree panels", 1); // tree structure for source points
		view_int_host point_leaf_panel_target ("target point leaf panel indices", point_count); // index of the target leaf panel containing this target point
		view_intt_host panel_points_inside_target ("target leaf panels contained points", 1, 1); // for target leaf panels, contains the indices of the contained target points
		view_int_host point_leaf_panel_source ("source point leaf panel indices", point_count); // index of the source leaf panel containing this source point
		view_intt_host panel_points_inside_source ("source leaf panels contained points", 1, 1); // for source leaf panels, contains the indices of the contained source points

		blfmm_tree_construction(run_config, tree_info_target, x_co_t, y_co_t, blfmm_panels_target, point_leaf_panel_target, panel_points_inside_target);
		blfmm_tree_construction(run_config, tree_info_source, x_co_s, y_co_s, blfmm_panels_source, point_leaf_panel_source, panel_points_inside_source);

		end = std::chrono::steady_clock::now();
		std::cout << "tree construction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		std::cout << "target blfmm panels: " << tree_info_target.panel_count << std::endl;
		std::cout << "target blfmm levels: " << tree_info_target.levels << std::endl;
		std::cout << "source blfmm panels: " << tree_info_source.panel_count << std::endl;
		std::cout << "source blfmm levels: " << tree_info_source.levels << std::endl;
		begin = std::chrono::steady_clock::now();

		view_interact_host interaction_list ("blfmm interactions", 1); // list of interactions

		dual_tree_traversal_2d(run_config, blfmm_panels_target, blfmm_panels_source, interaction_list);

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

		view_real d_x_co_t ("device target x coordiantes", point_count);
		view_real d_y_co_t ("device target y coordiantes", point_count);
		view_real d_x_co_s ("device source x coordiantes", point_count);
		view_real d_y_co_s ("device source y coordiantes", point_count);
		view_real d_charges ("device charges", point_count);
		view_real d_sol ("device solution", point_count);
		view_interact d_pp_ints ("device pp interactions", run_config.fmm_pp_count);
		view_interact d_pc_ints ("device pc interactions", run_config.fmm_pc_count);
		view_interact d_cp_ints ("device cp interactions", run_config.fmm_cp_count);
		view_interact d_cc_ints ("device cc interactions", run_config.fmm_cc_count);
		view_panel_2d d_blfmm_panels_target ("device target blfmm panels", tree_info_target.panel_count);
		view_panel_2d d_blfmm_panels_source ("device source blfmm panels", tree_info_source.panel_count);
		view_int d_point_leaf_panel_target ("device target point leaf panel indices", point_count);
		view_int d_point_leaf_panel_source ("device source point leaf panel indices", point_count);
		view_intt d_panel_points_inside_target ("device target leaf panels contained points", tree_info_target.panel_count, run_config.fmm_cluster_thresh);
		view_intt d_panel_points_inside_source ("device source leaf panels contained points", tree_info_source.panel_count, run_config.fmm_cluster_thresh);

		Kokkos::deep_copy(d_x_co_t, x_co_t);
		Kokkos::deep_copy(d_y_co_t, y_co_t);
		Kokkos::deep_copy(d_x_co_s, x_co_s);
		Kokkos::deep_copy(d_y_co_s, y_co_s);
		Kokkos::deep_copy(d_charges, charges);
		Kokkos::deep_copy(d_blfmm_panels_target, blfmm_panels_target);
		Kokkos::deep_copy(d_blfmm_panels_source, blfmm_panels_source);
		Kokkos::deep_copy(d_point_leaf_panel_target, point_leaf_panel_target);
		Kokkos::deep_copy(d_point_leaf_panel_source, point_leaf_panel_source);
		Kokkos::deep_copy(d_panel_points_inside_target, panel_points_inside_target);
		Kokkos::deep_copy(d_panel_points_inside_source, panel_points_inside_source);
		Kokkos::deep_copy(d_pp_ints, pp_interactions);
		Kokkos::deep_copy(d_pc_ints, pc_interactions);
		Kokkos::deep_copy(d_cp_ints, cp_interactions);
		Kokkos::deep_copy(d_cc_ints, cc_interactions);

		view_reall proxy_source_weights ("proxy source weights", tree_info_source.panel_count, run_config.interp_point_count);
		view_reall proxy_target_weights ("proxy target weights", tree_info_target.panel_count, run_config.interp_point_count);

		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info_source.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_source_weights, 0));
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {tree_info_target.panel_count, run_config.interp_point_count}), fill_2d<real>(proxy_target_weights, 0));
		Kokkos::parallel_for(point_count, fill_1d<real>(d_sol, 0));

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "host to device communication time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		upward_pass_2d(run_config, tree_info_source, d_x_co_s, d_y_co_s, d_charges, d_blfmm_panels_source, proxy_source_weights, d_point_leaf_panel_source);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "upward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		poisson_fmm_interactions_2d(run_config, d_x_co_t, d_y_co_t, d_x_co_s, d_y_co_s, d_charges, d_sol, d_panel_points_inside_source, d_panel_points_inside_target, proxy_source_weights, proxy_target_weights, d_pp_ints, d_pc_ints, d_cp_ints, d_cc_ints, d_blfmm_panels_source, d_blfmm_panels_target);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "interaction time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		downward_pass_2d(run_config, tree_info_target, d_x_co_t, d_y_co_t, d_sol, d_blfmm_panels_target, proxy_target_weights, d_panel_points_inside_target);

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "downward pass time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		// next, do the direct sum to check for accuracy
		view_real d_sol_ds ("device direct sum solution", point_count);
		Kokkos::parallel_for(point_count, fill_1d<real>(d_sol_ds, 0));
		Kokkos::parallel_for(Kokkos::MDRangePolicy({0, 0}, {point_count, point_count}), poisson_2d_dir_sum(d_x_co_t, d_y_co_t, d_x_co_s, d_y_co_s, d_charges, d_sol_ds, run_config.ker_eps));

		Kokkos::fence();
		end = std::chrono::steady_clock::now();
		std::cout << "direct sum time: " << std::chrono::duration<double>(end - begin).count() << " seconds" << std::endl;
		begin = std::chrono::steady_clock::now();

		Kokkos::deep_copy(sols, d_sol);
		Kokkos::deep_copy(sols_ds, d_sol_ds);

		real e1 = 0;
		real e2 = 0;

		for (int i = 0; i < point_count; i++) {
			e1 += (sols(i) - sols_ds(i)) * (sols(i) - sols_ds(i));
			e2 += sols_ds(i) * sols_ds(i);
		}

		std::cout << "relative l2 error is " << sqrt(e1/e2) << std::endl;
	}
	Kokkos::finalize();
	return 0;
}
