#include <Kokkos_Core.hpp>
#include <iostream>

#include "structs.hpp"
#include "barytreek-config.h"
#include "poisson_fmm_impl.hpp"

void poisson_fmm_interactions(const RunConfig& run_config, view_real& xcos, view_real& ycos, view_real& charges, view_real& soln, view_intt& panel_points_inside, 
								view_reall& proxy_source_weights, view_reall& proxy_target_weights, view_interact& pp_ints, view_interact& pc_ints, view_interact& cp_ints, view_interact& cc_ints, view_panel_2d& blfmm_panels) {
	// first compute list of interactions to compute
	int ints[run_config.mpi_p], lbs[run_config.mpi_p], ubs[run_config.mpi_p];
	for (int i = 0; i < run_config.mpi_p; i++) {
		ints[i] = int(run_config.fmm_pp_count / run_config.mpi_p);
	} 
	int total = run_config.mpi_p * ints[0];
	int gap = run_config.fmm_pp_count - total;
	for (int i = 1; i < gap + 1; i++) {
		ints[i] += 1;
	}
	lbs[0] = 0;
	ubs[0] = ints[0];
	for (int i = 1; i < run_config.mpi_p; i++) {
		lbs[i] = ubs[i-1];
		ubs[i] = lbs[i] + ints[i];
	}
	int lb, ub;
	lb = lbs[run_config.mpi_id];
	ub = ubs[run_config.mpi_id];

	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_pp_interaction(xcos, ycos, charges, soln, panel_points_inside, pp_ints, blfmm_panels, run_config.ker_eps));

	for (int i = 0; i < run_config.mpi_p; i++) {
		ints[i] = int(run_config.fmm_pc_count / run_config.mpi_p);
	} 
	total = run_config.mpi_p * ints[0];
	gap = run_config.fmm_pc_count - total;
	for (int i = 1; i < gap + 1; i++) {
		ints[i] += 1;
	}
	lbs[0] = 0;
	ubs[0] = ints[0];
	for (int i = 1; i < run_config.mpi_p; i++) {
		lbs[i] = ubs[i-1];
		ubs[i] = lbs[i] + ints[i];
	}
	lb = lbs[run_config.mpi_id];
	ub = ubs[run_config.mpi_id];

	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_pc_interaction(xcos, ycos, soln, panel_points_inside, proxy_source_weights, pc_ints, blfmm_panels, run_config.ker_eps, run_config.interp_degree));
	for (int i = 0; i < run_config.mpi_p; i++) {
		ints[i] = int(run_config.fmm_cp_count / run_config.mpi_p);
	} 
	total = run_config.mpi_p * ints[0];
	gap = run_config.fmm_cp_count - total;
	for (int i = 1; i < gap + 1; i++) {
		ints[i] += 1;
	}
	lbs[0] = 0;
	ubs[0] = ints[0];
	for (int i = 1; i < run_config.mpi_p; i++) {
		lbs[i] = ubs[i-1];
		ubs[i] = lbs[i] + ints[i];
	}
	lb = lbs[run_config.mpi_id];
	ub = ubs[run_config.mpi_id];

	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_cp_interaction(xcos, ycos, charges, panel_points_inside, proxy_target_weights, cp_ints, blfmm_panels, run_config.ker_eps, run_config.interp_degree));

	for (int i = 0; i < run_config.mpi_p; i++) {
		ints[i] = int(run_config.fmm_cc_count / run_config.mpi_p);
	} 
	total = run_config.mpi_p * ints[0];
	gap = run_config.fmm_cc_count - total;
	for (int i = 1; i < gap + 1; i++) {
		ints[i] += 1;
	}
	lbs[0] = 0;
	ubs[0] = ints[0];
	for (int i = 1; i < run_config.mpi_p; i++) {
		lbs[i] = ubs[i-1];
		ubs[i] = lbs[i] + ints[i];
	}
	lb = lbs[run_config.mpi_id];
	ub = ubs[run_config.mpi_id];
	Kokkos::parallel_for(Kokkos::RangePolicy(lb, ub), poisson_cc_interaction(proxy_target_weights, proxy_source_weights, cc_ints, blfmm_panels, run_config.ker_eps, run_config.interp_degree));
	Kokkos::fence();
}