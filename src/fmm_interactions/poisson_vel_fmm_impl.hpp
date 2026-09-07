#ifndef H_FMM_POISSON_VEL_IMPL_H
#define H_FMM_POISSON_VEL_IMPL_H

#include <Kokkos_Core.hpp>
#include "barytreek-config.h"
#include "structs.hpp"
#include <numbers>
#include "bli_impl.hpp"

struct poisson_vel_pp_interaction_3d {
	view_real xcos_t;
	view_real ycos_t;
	view_real zcos_t;
	view_real xcos_s;
	view_real ycos_s;
	view_real zcos_s;
	view_real charges;
	view_real vel_x;
	view_real vel_y;
	view_real vel_z;
	view_intt panel_points_inside_target;
	view_intt panel_points_inside_source;
	view_interact interaction_list;
	view_panel_3d blfmm_panels_target;
	view_panel_3d blfmm_panels_source;
	real eps;

	poisson_vel_pp_interaction_3d(view_real& xcos_t_, view_real& ycos_t_, view_real& zcos_t_, view_real& xcos_s_, view_real& ycos_s_, view_real& zcos_s_, view_real& charges_, 
							view_real& vel_x_, view_real& vel_y_, view_real& vel_z_, view_intt& panel_points_inside_target_, view_intt& panel_points_inside_source_, 
							view_interact& interactions_, view_panel_3d& blfmm_panels_target_, view_panel_3d& blfmm_panels_source_, real eps_) :
							xcos_t(xcos_t_), ycos_t(ycos_t_), zcos_t(zcos_t_), xcos_s(xcos_s_), ycos_s(ycos_s_), zcos_s(zcos_s_), charges(charges_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_),  
							panel_points_inside_target(panel_points_inside_target_), panel_points_inside_source(panel_points_inside_source_), 
							interaction_list(interactions_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int target_count = blfmm_panels_target(target_panel).point_count;
		int source_count = blfmm_panels_source(source_panel).point_count;
		real tx, ty, tz, sx, sy, sz, gfv, i_t, i_s;
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		for (int j = 0; j < target_count; j++) {
			i_t = panel_points_inside_target(target_panel,j);
			tx = xcos_t(i_t);
			ty = ycos_t(i_t);
			tz = zcos_t(i_t);
			for (int k = 0; k < source_count; k++) {
				i_s = panel_points_inside_source(source_panel,k);
				sx = xcos_s(i_s);
				sy = ycos_s(i_s);
				sz = zcos_s(i_s);
				gfv = charges(i_s)*gfc/((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps);
				Kokkos::atomic_add(&vel_x(i_t), (tx-sx)*gfv);
				Kokkos::atomic_add(&vel_y(i_t), (ty-sy)*gfv);
				Kokkos::atomic_add(&vel_z(i_t), (tz-sz)*gfv);
			}
		}
	}
};

struct poisson_vel_pc_interaction_3d {
	view_real xcos_t;
	view_real ycos_t;
	view_real zcos_t;
	view_real vel_x;
	view_real vel_y;
	view_real vel_z;
	view_intt panel_points_inside_target;
	view_reall proxy_source_weights;
	view_interact interaction_list;
	view_panel_3d blfmm_panels_target;
	view_panel_3d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_vel_pc_interaction_3d(view_real& xcos_t_, view_real& ycos_t_, view_real& zcos_t_, view_real& vel_x_, view_real& vel_y_, view_real& vel_z_, view_intt& panel_points_inside_target_, 
							view_reall& proxy_source_weights_, view_interact& interaction_list_, view_panel_3d& blfmm_panels_target_, 
							view_panel_3d& blfmm_panels_source_, real eps_, int interp_deg_) : xcos_t(xcos_t_), ycos_t(ycos_t_), zcos_t(zcos_t_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_),  
							panel_points_inside_target(panel_points_inside_target_), proxy_source_weights(proxy_source_weights_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), 
							eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int target_count = blfmm_panels_target(target_panel).point_count;
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		real min_x, max_x, x, min_y, max_y, y, min_z, max_z, z, cheb_x[max_degree+1], cheb_y[max_degree+1], cheb_z[max_degree+1], tx, ty, tz, sx, sy, sz, gfv;
		min_x = blfmm_panels_source(source_panel).min_x;
		max_x = blfmm_panels_source(source_panel).max_x;
		min_y = blfmm_panels_source(source_panel).min_y;
		max_y = blfmm_panels_source(source_panel).max_y;
		min_z = blfmm_panels_source(source_panel).min_z;
		max_z = blfmm_panels_source(source_panel).max_z;
		bli_points_shift(cheb_x, min_x, max_x, interp_deg);
		bli_points_shift(cheb_y, min_y, max_y, interp_deg);
		bli_points_shift(cheb_z, min_z, max_z, interp_deg);
		int i_t, index;
		for (int l = 0; l < target_count; l++) {
			i_t = panel_points_inside_target(target_panel,l);
			tx = xcos_t(i_t);
			ty = ycos_t(i_t);
			tz = zcos_t(i_t);
			index = 0;
			for (int j = 0; j < interp_deg+1; j++) { // x loop
				sx = cheb_x[j];
				for (int k = 0; k < interp_deg+1; k++) { // y loop
					sy = cheb_y[k];
					for (int l = 0; l < interp_deg+1; l++) { // z loop
						sz = cheb_z[l];
						gfv = proxy_source_weights(source_panel,index)*gfc/((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps);
						Kokkos::atomic_add(&vel_x(i_t), (tx-sx)*gfv);
						Kokkos::atomic_add(&vel_y(i_t), (ty-sy)*gfv);
						Kokkos::atomic_add(&vel_z(i_t), (tz-sz)*gfv);
						index += 1;
					}
				}
			}
		}
	}
};

struct poisson_vel_cp_interaction_3d {
	view_real xcos_s;
	view_real ycos_s;
	view_real zcos_s;
	view_real charges;
	view_intt panel_points_inside_source;
	view_reall proxy_target_weights_x;
	view_reall proxy_target_weights_y;
	view_reall proxy_target_weights_z;
	view_interact interaction_list;
	view_panel_3d blfmm_panels_target;
	view_panel_3d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_vel_cp_interaction_3d(view_real& xcos_s_, view_real& ycos_s_, view_real& zcos_s_, view_real& charges_, view_intt& panel_points_inside_source_, 
							view_reall& proxy_target_weights_x_, view_reall& proxy_target_weights_y_, view_reall& proxy_target_weights_z_, view_interact& interaction_list_, view_panel_3d& blfmm_panels_target_, 
							view_panel_3d& blfmm_panels_source_, real eps_, int interp_deg_) : xcos_s(xcos_s_), ycos_s(ycos_s_), zcos_s(zcos_s_), charges(charges_), 
							panel_points_inside_source(panel_points_inside_source_), proxy_target_weights_x(proxy_target_weights_x_), proxy_target_weights_y(proxy_target_weights_y_), proxy_target_weights_z(proxy_target_weights_z_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), blfmm_panels_source(blfmm_panels_source_), 
							eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		int source_count = blfmm_panels_source(source_panel).point_count;
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		real min_x, max_x, x, min_y, max_y, y, min_z, max_z, z, cheb_x[max_degree+1], cheb_y[max_degree+1], cheb_z[max_degree+1], tx, ty, tz, sx, sy, sz, gfv;
		min_x = blfmm_panels_target(target_panel).min_x;
		max_x = blfmm_panels_target(target_panel).max_x;
		min_y = blfmm_panels_target(target_panel).min_y;
		max_y = blfmm_panels_target(target_panel).max_y;
		min_z = blfmm_panels_target(target_panel).min_z;
		max_z = blfmm_panels_target(target_panel).max_z;
		bli_points_shift(cheb_x, min_x, max_x, interp_deg);
		bli_points_shift(cheb_y, min_y, max_y, interp_deg);
		bli_points_shift(cheb_z, min_z, max_z, interp_deg);
		int i_s, index = 0;
		for (int j = 0; j < interp_deg+1; j++) { // x loop
			tx = cheb_x[j];
			for (int k = 0; k < interp_deg+1; k++) { // y loop
				ty = cheb_y[k];
				for (int l = 0; l < interp_deg+1; l++) { // z loop
					tz = cheb_z[l];
					for (int m = 0; m < source_count; m++) {
						i_s = panel_points_inside_source(source_panel, m);
						sx = xcos_s(i_s);
						sy = ycos_s(i_s);
						sz = zcos_s(i_s);
						gfv = charges(i_s)*gfc/((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps);
						Kokkos::atomic_add(&proxy_target_weights_x(target_panel,index), (tx-sx)*gfv);
						Kokkos::atomic_add(&proxy_target_weights_y(target_panel,index), (ty-sy)*gfv);
						Kokkos::atomic_add(&proxy_target_weights_z(target_panel,index), (tz-sz)*gfv);
					}
					index += 1;
				}
			}
		}
	}
};

struct poisson_vel_cc_interaction_3d {
	view_reall proxy_target_weights_x;
	view_reall proxy_target_weights_y;
	view_reall proxy_target_weights_z;
	view_reall proxy_source_weights;
	view_interact interaction_list;
	view_panel_3d blfmm_panels_target;
	view_panel_3d blfmm_panels_source;
	real eps;
	int interp_deg;

	poisson_vel_cc_interaction_3d(view_reall& proxy_target_weights_x_, view_reall& proxy_target_weights_y_, view_reall& proxy_target_weights_z_, view_reall& proxy_source_weights_, view_interact& interaction_list_, 
							view_panel_3d& blfmm_panels_target_, view_panel_3d& blfmm_panels_source_, real eps_, int interp_deg_) :
							proxy_target_weights_x(proxy_target_weights_x_), proxy_target_weights_y(proxy_target_weights_y_), proxy_target_weights_z(proxy_target_weights_z_), proxy_source_weights(proxy_source_weights_), 
							interaction_list(interaction_list_), blfmm_panels_target(blfmm_panels_target_), 
							blfmm_panels_source(blfmm_panels_source_), eps(eps_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_panel = interaction_list(i).target_panel;
		int source_panel = interaction_list(i).source_panel;
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		real min_x_t = blfmm_panels_target(target_panel).min_x;
		real max_x_t = blfmm_panels_target(target_panel).max_x;
		real min_y_t = blfmm_panels_target(target_panel).min_y;
		real max_y_t = blfmm_panels_target(target_panel).max_y;
		real min_z_t = blfmm_panels_target(target_panel).min_z;
		real max_z_t = blfmm_panels_target(target_panel).max_z;
		real min_x_s = blfmm_panels_source(source_panel).min_x;
		real max_x_s = blfmm_panels_source(source_panel).max_x;
		real min_y_s = blfmm_panels_source(source_panel).min_y;
		real max_y_s = blfmm_panels_source(source_panel).max_y;
		real min_z_s = blfmm_panels_source(source_panel).min_z;
		real max_z_s = blfmm_panels_source(source_panel).max_z;
		real tx, ty, tz, sx, sy, sz, gfv;
		real cheb_x_t[max_degree+1], cheb_y_t[max_degree+1], cheb_z_t[max_degree+1], cheb_x_s[max_degree+1], cheb_y_s[max_degree+1], cheb_z_s[max_degree+1];
		bli_points_shift(cheb_x_t, min_x_t, max_x_t, interp_deg);
		bli_points_shift(cheb_y_t, min_y_t, max_y_t, interp_deg);
		bli_points_shift(cheb_z_t, min_z_t, max_z_t, interp_deg);
		bli_points_shift(cheb_x_s, min_x_s, max_x_s, interp_deg);
		bli_points_shift(cheb_y_s, min_y_s, max_y_s, interp_deg);
		bli_points_shift(cheb_z_s, min_z_s, max_z_s, interp_deg);
		int index_t, index_s;
		index_t = 0;
		for (int j1 = 0; j1 < interp_deg+1; j1++) { // target x loop
			tx = cheb_x_t[j1];
			for (int k1 = 0; k1 < interp_deg+1; k1++) { // target y loop
				ty = cheb_y_t[k1];
				for (int l1 = 0; l1 < interp_deg+1; l1++) { // target z loop
					tz = cheb_z_t[l1];
					index_s = 0;
					for (int j2 = 0; j2 < interp_deg+1; j2++) { // source x loop
						sx = cheb_x_s[j2];
						for (int k2 = 0; k2 < interp_deg+1; k2++) { // source y loop
							sy = cheb_y_s[k2];
							for (int l2 = 0; l2 < interp_deg+1; l2++) { // source z loop
								sz = cheb_z_s[l2];
								gfv = proxy_source_weights(source_panel,index_s)*gfc/((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps);
								Kokkos::atomic_add(&proxy_target_weights_x(target_panel,index_t), (tx-sx)*gfv);
								Kokkos::atomic_add(&proxy_target_weights_y(target_panel,index_t), (ty-sy)*gfv);
								Kokkos::atomic_add(&proxy_target_weights_z(target_panel,index_t), (tz-sz)*gfv);
								index_s += 1;
							}
						}
					}
					index_t += 1;
				}
			}
		}
	}
};

#endif