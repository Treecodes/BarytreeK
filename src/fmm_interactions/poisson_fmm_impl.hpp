#ifndef H_FMM_POISSON_IMPL_H
#define H_FMM_POISSON_IMPL_H

#include <Kokkos_Core.hpp>
#include "barytreek-config.h"
#include "structs.hpp"
#include <numbers>
#include "bli_impl.hpp"

struct poisson_pp_interaction_2d {
	view_real xcos_t;
	view_real ycos_t;
	view_real xcos_s;
	view_real ycos_s;
	view_real charges;
	view_real soln;
	view_intt panel_points_inside_target;
	view_intt panel_points_inside_source;
	view_interact interaction_list;
	view_panel_2d blfmm_panels_target;
	view_panel_2d blfmm_panels_source;
	real eps;

	poisson_pp_interaction_2d(view_real& xcos_t_, view_real& ycos_t_, view_real& xcos_s_, view_real& ycos_s_, view_real& charges_, 
							view_real& soln_, view_intt& panel_points_inside_target_, view_intt& panel_points_inside_source_, 
							view_interact& interactions_, view_panel_2d& blfmm_panels_target_, view_panel_2d& blfmm_panels_source_, real eps_) :
							xcos_t(xcos_t_), ycos_t(ycos_t_), xcos_s(xcos_s_), ycos_s(ycos_s_), charges(charges_), soln(soln_), 
							panel_points_inside_target(panel_points_inside_target_), panel_points_inside_source(panel_points_inside_source_), 
							interaction_list(interactions_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int target_count = blfmm_panels_target(target_panel).point_count;
		int source_count = blfmm_panels_source(source_panel).point_count;
		real tx, ty, sx, sy, gfv, i_t, i_s;
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		for (int j = 0; j < target_count; j++) {
			i_t = panel_points_inside_target(target_panel,j);
			tx = xcos_t(i_t);
			ty = ycos_t(i_t);
			for (int k = 0; k < source_count; k++) {
				i_s = panel_points_inside_source(source_panel,k);
				sx = xcos_s(i_s);
				sy = ycos_s(i_s);
				gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
				Kokkos::atomic_add(&soln(i_t), gfv*charges(i_s));
			}
		}
	}
};

struct poisson_pc_interaction_2d {
	view_real xcos_t;
	view_real ycos_t;
	view_real soln;
	view_intt panel_points_inside_target;
	view_reall proxy_source_weights;
	view_interact interaction_list;
	view_panel_2d blfmm_panels_target;
	view_panel_2d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_pc_interaction_2d(view_real& xcos_t_, view_real& ycos_t_, view_real& soln_, view_intt& panel_points_inside_target_, 
							view_reall& proxy_source_weights_, view_interact& interaction_list_, view_panel_2d& blfmm_panels_target_, 
							view_panel_2d& blfmm_panels_source_, real eps_, int interp_deg_) : xcos_t(xcos_t_), ycos_t(ycos_t_), soln(soln_), 
							panel_points_inside_target(panel_points_inside_target_), proxy_source_weights(proxy_source_weights_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), 
							eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int target_count = blfmm_panels_target(target_panel).point_count;
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		real min_x, max_x, x, min_y, max_y, y, cheb_x[max_degree+1], cheb_y[max_degree+1], tx, ty, sx, sy, gfv;
		min_x = blfmm_panels_source(source_panel).min_x;
		max_x = blfmm_panels_source(source_panel).max_x;
		min_y = blfmm_panels_source(source_panel).min_y;
		max_y = blfmm_panels_source(source_panel).max_y;
		bli_points_shift(cheb_x, min_x, max_x, interp_deg);
		bli_points_shift(cheb_y, min_y, max_y, interp_deg);
		int i_t, index;
		for (int l = 0; l < target_count; l++) {
			i_t = panel_points_inside_target(target_panel,l);
			tx = xcos_t(i_t);
			ty = ycos_t(i_t);
			index = 0;
			for (int j = 0; j < interp_deg+1; j++) { // x loop
				for (int k = 0; k < interp_deg+1; k++) { // y loop
					sx = cheb_x[j];
					sy = cheb_y[k];
					gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
					Kokkos::atomic_add(&soln(i_t), gfv*proxy_source_weights(source_panel,index));
					index += 1;
				}
			}
		}
	}
};

struct poisson_cp_interaction_2d {
	view_real xcos_s;
	view_real ycos_s;
	view_real charges;
	view_intt panel_points_inside_source;
	view_reall proxy_target_weights;
	view_interact interaction_list;
	view_panel_2d blfmm_panels_target;
	view_panel_2d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_cp_interaction_2d(view_real& xcos_s_, view_real& ycos_s_, view_real& charges_, view_intt& panel_points_inside_source_, 
							view_reall& proxy_target_weights_, view_interact& interaction_list_, view_panel_2d& blfmm_panels_target_, 
							view_panel_2d& blfmm_panels_source_, real eps_, int interp_deg_) : xcos_s(xcos_s_), ycos_s(ycos_s_), charges(charges_), 
							panel_points_inside_source(panel_points_inside_source_), proxy_target_weights(proxy_target_weights_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), 
							eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int source_count = blfmm_panels_source(source_panel).point_count;
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		real min_x, max_x, x, min_y, max_y, y, cheb_x[max_degree+1], cheb_y[max_degree+1], tx, ty, sx, sy, gfv;
		min_x = blfmm_panels_target(target_panel).min_x;
		max_x = blfmm_panels_target(target_panel).max_x;
		min_y = blfmm_panels_target(target_panel).min_y;
		max_y = blfmm_panels_target(target_panel).max_y;
		bli_points_shift(cheb_x, min_x, max_x, interp_deg);
		bli_points_shift(cheb_y, min_y, max_y, interp_deg);
		int i_s, index = 0;
		for (int j = 0; j < interp_deg+1; j++) { // x loop
			for (int k = 0; k < interp_deg+1; k++) { // y loop
				tx = cheb_x[j];
				ty = cheb_y[k];
				for (int l = 0; l < source_count; l++) {
					i_s = panel_points_inside_source(source_panel,l);
					sx = xcos_s(i_s);
					sy = ycos_s(i_s);
					gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
					Kokkos::atomic_add(&proxy_target_weights(target_panel,index), gfv*charges(i_s));
				}
				index += 1;
			}
		}
	}
};

struct poisson_cc_interaction_2d {
	view_reall proxy_target_weights;
	view_reall proxy_source_weights;
	view_interact interaction_list;
	view_panel_2d blfmm_panels_target;
	view_panel_2d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_cc_interaction_2d(view_reall& proxy_target_weights_, view_reall& proxy_source_weights_, view_interact& interaction_list_, 
							view_panel_2d& blfmm_panels_target_, view_panel_2d& blfmm_panels_source_, real eps_, int interp_deg_) :
							proxy_target_weights(proxy_target_weights_), proxy_source_weights(proxy_source_weights_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), 
							blfmm_panels_source(blfmm_panels_source_), eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		real min_x_t = blfmm_panels_target(target_panel).min_x;
		real max_x_t = blfmm_panels_target(target_panel).max_x;
		real min_y_t = blfmm_panels_target(target_panel).min_y;
		real max_y_t = blfmm_panels_target(target_panel).max_y;
		real min_x_s = blfmm_panels_source(source_panel).min_x;
		real max_x_s = blfmm_panels_source(source_panel).max_x;
		real min_y_s = blfmm_panels_source(source_panel).min_y;
		real max_y_s = blfmm_panels_source(source_panel).max_y;
		real tx, ty, sx, sy, gfv;
		real cheb_x_t[max_degree+1], cheb_y_t[max_degree+1], cheb_x_s[max_degree+1], cheb_y_s[max_degree+1];
		bli_points_shift(cheb_x_t, min_x_t, max_x_t, interp_deg);
		bli_points_shift(cheb_y_t, min_y_t, max_y_t, interp_deg);
		bli_points_shift(cheb_x_s, min_x_s, max_x_s, interp_deg);
		bli_points_shift(cheb_y_s, min_y_s, max_y_s, interp_deg);
		int index_t, index_s;
		index_t = 0;
		for (int j1 = 0; j1 < interp_deg+1; j1++) { // target x loop
			for (int k1 = 0; k1 < interp_deg+1; k1++) { // target y loop
				tx = cheb_x_t[j1];
				ty = cheb_y_t[k1];
				index_s = 0;
				for (int j2 = 0; j2 < interp_deg+1; j2++) { // source x loop
					for (int k2 = 0; k2 < interp_deg+1; k2++) {	// source y loop
						sx = cheb_x_s[j2];
						sy = cheb_y_s[k2];
						gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
						Kokkos::atomic_add(&proxy_target_weights(target_panel,index_t), gfv*proxy_source_weights(source_panel,index_s));
						index_s += 1;
					}
				}
				index_t += 1;
			}
		}
	}
};

#endif