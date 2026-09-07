#ifndef H_DOWNWARD_PASS_IMPL_H
#define H_DOWNWARD_PASS_IMPL_H

#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "bli_impl.hpp"

struct parent_to_child_2d {
	view_panel_2d blfmm_panels;
	view_reall proxy_target_weights;
	int interp_deg;

	parent_to_child_2d(view_panel_2d& blfmm_panels_, view_reall& proxy_target_weights_, int interp_deg_) : 
					blfmm_panels(blfmm_panels_), proxy_target_weights(proxy_target_weights_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		// interpolates proxy target weights of panel i to child panels
		int child;
		switch (j) {
			case 0:
				child = blfmm_panels(i).child1;
				break;
			case 1:
				child = blfmm_panels(i).child2;
				break;
			case 2:
				child = blfmm_panels(i).child3;
				break;
			case 3:
				child = blfmm_panels(i).child4;
				break;
			default:
				throw std::runtime_error("j not between 0 and 3, parent to child");
		}
		real min_x_p = blfmm_panels(i).min_x;
		real max_x_p = blfmm_panels(i).max_x;
		real min_y_p = blfmm_panels(i).min_y;
		real max_y_p = blfmm_panels(i).max_y;
		real min_x_c = blfmm_panels(child).min_x;
		real max_x_c = blfmm_panels(child).max_x;
		real min_y_c = blfmm_panels(child).min_y;
		real max_y_c = blfmm_panels(child).max_y;
		real cheb_x[max_degree+1], cheb_y[max_degree+1], x_basis_vals[max_degree+1], y_basis_vals[max_degree+1];
		bli_points_shift(cheb_x, min_x_c, max_x_c, interp_deg);
		bli_points_shift(cheb_y, min_y_c, max_y_c, interp_deg);
		real x, y;
		int index1, index2;
		index1 = 0;
		for (int j = 0; j < interp_deg+1; j++) { // child x loop
			x = cheb_x[j];
			for (int k = 0; k < interp_deg+1; k++) { // child y loop
				y = cheb_y[k];
				interp_vals_bli(x_basis_vals, x, min_x_p, max_x_p, interp_deg);
				interp_vals_bli(y_basis_vals, y, min_y_p, max_y_p, interp_deg);
				index2 = 0;
				for (int j2 = 0; j2 < interp_deg+1; j2++) {
					for (int k2 = 0; k2 < interp_deg+1; k2++) {
						Kokkos::atomic_add(&proxy_target_weights(child,index1), x_basis_vals[j2]*y_basis_vals[k2]*proxy_target_weights(i,index2));
						index2 += 1;
					}
				}
				index1 += 1;
			}
		}
	}
};

struct leaf_to_point_2d {
	view_real xcos;
	view_real ycos;
	view_real soln;
	view_reall proxy_target_weights;
	view_panel_2d blfmm_panels;
	view_intt panel_points_inside;
	int interp_deg;

	leaf_to_point_2d(view_real& xcos_, view_real& ycos_, view_real& soln_, view_reall& proxy_target_weights_, view_panel_2d& blfmm_panels_, view_intt& panel_points_inside_, int interp_deg_) :
					xcos(xcos_), ycos(ycos_), soln(soln_), proxy_target_weights(proxy_target_weights_), blfmm_panels(blfmm_panels_), panel_points_inside(panel_points_inside_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_count = blfmm_panels(i).point_count;
		real min_x = blfmm_panels(i).min_x;
		real max_x = blfmm_panels(i).max_x;
		real min_y = blfmm_panels(i).min_y;
		real max_y = blfmm_panels(i).max_y;
		int i_t;
		real x, y, x_basis_vals[max_degree+1], y_basis_vals[max_degree+1];
		int index;
		for (int j = 0; j < target_count; j++) {
			i_t = panel_points_inside(i,j);
			x = xcos(i_t);
			y = ycos(i_t);
			interp_vals_bli(x_basis_vals, x, min_x, max_x, interp_deg);
			interp_vals_bli(y_basis_vals, y, min_y, max_y, interp_deg);
			index = 0;
			for (int k = 0; k < interp_deg+1; k++) {
				for (int l = 0; l < interp_deg+1; l++) {
					Kokkos::atomic_add(&soln(i_t), proxy_target_weights(i,index) * x_basis_vals[k] * y_basis_vals[l]);
					index += 1;
				}
			}
		}
	}
};

struct parent_to_child_3d {
	view_panel_3d blfmm_panels;
	view_reall proxy_target_weights;
	int interp_deg;

	parent_to_child_3d(view_panel_3d& blfmm_panels_, view_reall& proxy_target_weights_, int interp_deg_) : 
					blfmm_panels(blfmm_panels_), proxy_target_weights(proxy_target_weights_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		// interpolates proxy target weights of panel i to child panels
		int child;
		switch (j) {
			case 0:
				child = blfmm_panels(i).child1;
				break;
			case 1:
				child = blfmm_panels(i).child2;
				break;
			case 2:
				child = blfmm_panels(i).child3;
				break;
			case 3:
				child = blfmm_panels(i).child4;
				break;
			case 4:
				child = blfmm_panels(i).child5;
				break;
			case 5:
				child = blfmm_panels(i).child6;
				break;
			case 6:
				child = blfmm_panels(i).child7;
				break;
			case 7:
				child = blfmm_panels(i).child8;
				break;
			default:
				throw std::runtime_error("j not between 0 and 3, parent to child");
		}
		real min_x_p = blfmm_panels(i).min_x;
		real max_x_p = blfmm_panels(i).max_x;
		real min_y_p = blfmm_panels(i).min_y;
		real max_y_p = blfmm_panels(i).max_y;
		real min_z_p = blfmm_panels(i).min_z;
		real max_z_p = blfmm_panels(i).max_z;
		real min_x_c = blfmm_panels(child).min_x;
		real max_x_c = blfmm_panels(child).max_x;
		real min_y_c = blfmm_panels(child).min_y;
		real max_y_c = blfmm_panels(child).max_y;
		real min_z_c = blfmm_panels(child).min_z;
		real max_z_c = blfmm_panels(child).max_z;
		real cheb_x[max_degree+1], cheb_y[max_degree+1], cheb_z[max_degree+1], x_basis_vals[max_degree+1], y_basis_vals[max_degree+1], z_basis_vals[max_degree+1];
		bli_points_shift(cheb_x, min_x_c, max_x_c, interp_deg);
		bli_points_shift(cheb_y, min_y_c, max_y_c, interp_deg);
		bli_points_shift(cheb_z, min_z_c, max_z_c, interp_deg);
		real x, y, z;
		int index1, index2;
		index1 = 0;
		for (int j = 0; j < interp_deg+1; j++) { // child x loop
			x = cheb_x[j];
			interp_vals_bli(x_basis_vals, x, min_x_p, max_x_p, interp_deg);
			for (int k = 0; k < interp_deg+1; k++) { // child y loop
				y = cheb_y[k];
				interp_vals_bli(y_basis_vals, y, min_y_p, max_y_p, interp_deg);
				for (int l = 0; l < interp_deg+1; l++) { // child z loop
					z = cheb_z[l];
					interp_vals_bli(z_basis_vals, z, min_z_p, max_z_p, interp_deg);
					index2 = 0;
					for (int j2 = 0; j2 < interp_deg+1; j2++) {
						for (int k2 = 0; k2 < interp_deg+1; k2++) {
							for (int l2 = 0; l2 < interp_deg+1; l2++) {
								Kokkos::atomic_add(&proxy_target_weights(child,index1), x_basis_vals[j2]*y_basis_vals[k2]*z_basis_vals[l2]*proxy_target_weights(i,index2));
								index2 += 1;
							}
						}
					}
					index1 += 1;
				}
			}
		}
	}
};

struct leaf_to_point_3d {
	view_real xcos;
	view_real ycos;
	view_real zcos;
	view_real soln;
	view_reall proxy_target_weights;
	view_panel_3d blfmm_panels;
	view_intt panel_points_inside;
	int interp_deg;

	leaf_to_point_3d(view_real& xcos_, view_real& ycos_, view_real& zcos_, view_real& soln_, view_reall& proxy_target_weights_, view_panel_3d& blfmm_panels_, view_intt& panel_points_inside_, int interp_deg_) :
					xcos(xcos_), ycos(ycos_), zcos(zcos_), soln(soln_), proxy_target_weights(proxy_target_weights_), blfmm_panels(blfmm_panels_), panel_points_inside(panel_points_inside_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_count = blfmm_panels(i).point_count;
		real min_x = blfmm_panels(i).min_x;
		real max_x = blfmm_panels(i).max_x;
		real min_y = blfmm_panels(i).min_y;
		real max_y = blfmm_panels(i).max_y;
		real min_z = blfmm_panels(i).min_z;
		real max_z = blfmm_panels(i).max_z;
		int i_t;
		real x, y, z, x_basis_vals[max_degree+1], y_basis_vals[max_degree+1], z_basis_vals[max_degree+1];
		int index;
		for (int j = 0; j < target_count; j++) {
			i_t = panel_points_inside(i,j);
			x = xcos(i_t);
			y = ycos(i_t);
			z = zcos(i_t);
			interp_vals_bli(x_basis_vals, x, min_x, max_x, interp_deg);
			interp_vals_bli(y_basis_vals, y, min_y, max_y, interp_deg);
			interp_vals_bli(z_basis_vals, z, min_z, max_z, interp_deg);
			index = 0;
			for (int k = 0; k < interp_deg+1; k++) { // x loop
				for (int l = 0; l < interp_deg+1; l++) { // y loop
					for (int m = 0; m < interp_deg+1; m++) { // z loop
						Kokkos::atomic_add(&soln(i_t), proxy_target_weights(i,index) * x_basis_vals[k] * y_basis_vals[l] * z_basis_vals[m]);
						index += 1;
					}
				}
			}
		}
	}
};

struct parent_to_child_3d_3 {
	view_panel_3d blfmm_panels;
	view_reall proxy_target_weights_x;
	view_reall proxy_target_weights_y;
	view_reall proxy_target_weights_z;
	int interp_deg;

	parent_to_child_3d_3(view_panel_3d& blfmm_panels_, view_reall& proxy_target_weights_x_, view_reall& proxy_target_weights_y_, view_reall& proxy_target_weights_z_, int interp_deg_) : 
					blfmm_panels(blfmm_panels_), proxy_target_weights_x(proxy_target_weights_x_), proxy_target_weights_y(proxy_target_weights_y_), proxy_target_weights_z(proxy_target_weights_z_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		// interpolates proxy target weights of panel i to child panels
		int child;
		switch (j) {
			case 0:
				child = blfmm_panels(i).child1;
				break;
			case 1:
				child = blfmm_panels(i).child2;
				break;
			case 2:
				child = blfmm_panels(i).child3;
				break;
			case 3:
				child = blfmm_panels(i).child4;
				break;
			case 4:
				child = blfmm_panels(i).child5;
				break;
			case 5:
				child = blfmm_panels(i).child6;
				break;
			case 6:
				child = blfmm_panels(i).child7;
				break;
			case 7:
				child = blfmm_panels(i).child8;
				break;
			default:
				throw std::runtime_error("j not between 0 and 3, parent to child");
		}
		real min_x_p = blfmm_panels(i).min_x;
		real max_x_p = blfmm_panels(i).max_x;
		real min_y_p = blfmm_panels(i).min_y;
		real max_y_p = blfmm_panels(i).max_y;
		real min_z_p = blfmm_panels(i).min_z;
		real max_z_p = blfmm_panels(i).max_z;
		real min_x_c = blfmm_panels(child).min_x;
		real max_x_c = blfmm_panels(child).max_x;
		real min_y_c = blfmm_panels(child).min_y;
		real max_y_c = blfmm_panels(child).max_y;
		real min_z_c = blfmm_panels(child).min_z;
		real max_z_c = blfmm_panels(child).max_z;
		real cheb_x[max_degree+1], cheb_y[max_degree+1], cheb_z[max_degree+1], x_basis_vals[max_degree+1], y_basis_vals[max_degree+1], z_basis_vals[max_degree+1];
		bli_points_shift(cheb_x, min_x_c, max_x_c, interp_deg);
		bli_points_shift(cheb_y, min_y_c, max_y_c, interp_deg);
		bli_points_shift(cheb_z, min_z_c, max_z_c, interp_deg);
		real x, y, z;
		int index1, index2;
		index1 = 0;
		for (int j = 0; j < interp_deg+1; j++) { // child x loop
			x = cheb_x[j];
			interp_vals_bli(x_basis_vals, x, min_x_p, max_x_p, interp_deg);
			for (int k = 0; k < interp_deg+1; k++) { // child y loop
				y = cheb_y[k];
				interp_vals_bli(y_basis_vals, y, min_y_p, max_y_p, interp_deg);
				for (int l = 0; l < interp_deg+1; l++) { // child z loop
					z = cheb_z[l];
					interp_vals_bli(z_basis_vals, z, min_z_p, max_z_p, interp_deg);
					index2 = 0;
					for (int j2 = 0; j2 < interp_deg+1; j2++) {
						for (int k2 = 0; k2 < interp_deg+1; k2++) {
							for (int l2 = 0; l2 < interp_deg+1; l2++) {
								Kokkos::atomic_add(&proxy_target_weights_x(child,index1), x_basis_vals[j2]*y_basis_vals[k2]*z_basis_vals[l2]*proxy_target_weights_x(i,index2));
								Kokkos::atomic_add(&proxy_target_weights_y(child,index1), x_basis_vals[j2]*y_basis_vals[k2]*z_basis_vals[l2]*proxy_target_weights_y(i,index2));
								Kokkos::atomic_add(&proxy_target_weights_z(child,index1), x_basis_vals[j2]*y_basis_vals[k2]*z_basis_vals[l2]*proxy_target_weights_z(i,index2));
								index2 += 1;
							}
						}
					}
					index1 += 1;
				}
			}
		}
	}
};

struct leaf_to_point_3d_3 {
	view_real xcos;
	view_real ycos;
	view_real zcos;
	view_real vel_x;
	view_real vel_y;
	view_real vel_z;
	view_reall proxy_target_weights_x;
	view_reall proxy_target_weights_y;
	view_reall proxy_target_weights_z;
	view_panel_3d blfmm_panels;
	view_intt panel_points_inside;
	int interp_deg;

	leaf_to_point_3d_3(view_real& xcos_, view_real& ycos_, view_real& zcos_, view_real& vel_x_, view_real& vel_y_, view_real& vel_z_, 
						view_reall& proxy_target_weights_x_, view_reall& proxy_target_weights_y_, view_reall& proxy_target_weights_z_, view_panel_3d& blfmm_panels_, view_intt& panel_points_inside_, int interp_deg_) :
						xcos(xcos_), ycos(ycos_), zcos(zcos_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), proxy_target_weights_x(proxy_target_weights_x_), proxy_target_weights_y(proxy_target_weights_y_), proxy_target_weights_z(proxy_target_weights_z_),
						blfmm_panels(blfmm_panels_), panel_points_inside(panel_points_inside_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		int target_count = blfmm_panels(i).point_count;
		real min_x = blfmm_panels(i).min_x;
		real max_x = blfmm_panels(i).max_x;
		real min_y = blfmm_panels(i).min_y;
		real max_y = blfmm_panels(i).max_y;
		real min_z = blfmm_panels(i).min_z;
		real max_z = blfmm_panels(i).max_z;
		int i_t;
		real x, y, z, x_basis_vals[max_degree+1], y_basis_vals[max_degree+1], z_basis_vals[max_degree+1];
		int index;
		for (int j = 0; j < target_count; j++) {
			i_t = panel_points_inside(i,j);
			x = xcos(i_t);
			y = ycos(i_t);
			z = zcos(i_t);
			interp_vals_bli(x_basis_vals, x, min_x, max_x, interp_deg);
			interp_vals_bli(y_basis_vals, y, min_y, max_y, interp_deg);
			interp_vals_bli(z_basis_vals, z, min_z, max_z, interp_deg);
			index = 0;
			for (int k = 0; k < interp_deg+1; k++) { // x loop
				for (int l = 0; l < interp_deg+1; l++) { // y loop
					for (int m = 0; m < interp_deg+1; m++) { // z loop
						Kokkos::atomic_add(&vel_x(i_t), proxy_target_weights_x(i,index) * x_basis_vals[k] * y_basis_vals[l] * z_basis_vals[m]);
						Kokkos::atomic_add(&vel_y(i_t), proxy_target_weights_y(i,index) * x_basis_vals[k] * y_basis_vals[l] * z_basis_vals[m]);
						Kokkos::atomic_add(&vel_z(i_t), proxy_target_weights_z(i,index) * x_basis_vals[k] * y_basis_vals[l] * z_basis_vals[m]);
						index += 1;
					}
				}
			}
		}
	}
};

#endif