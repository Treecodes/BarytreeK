#ifndef H_UPWARD_PASS_IMPL_H
#define H_UPWARD_PASS_IMPL_H

#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "bli_impl.hpp"

struct base_pots_2d {
	view_real xcos;
	view_real ycos;
	view_real charges;
	view_panel_2d blfmm_panels;
	view_reall proxy_source_weights;
	view_int point_leaf_panel;
	int interp_deg;

	base_pots_2d(view_real& xcos_, view_real& ycos_, view_real& charges_, view_panel_2d& blfmm_panels_, 
				view_reall& proxy_source_weights_, view_int& point_leaf_panel_, int interp_deg_) :
				xcos(xcos_), ycos(ycos_), charges(charges_), blfmm_panels(blfmm_panels_), 
				proxy_source_weights(proxy_source_weights_), point_leaf_panel(point_leaf_panel_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// computes the contribution to the proxy source weights of source point i
		real x = xcos(i);
		real y = ycos(i);
		int leaf = point_leaf_panel(i);
		real x_basis_vals[max_degree+1], y_basis_vals[max_degree+1];
		real min_x = blfmm_panels(leaf).min_x;
		real max_x = blfmm_panels(leaf).max_x;
		real min_y = blfmm_panels(leaf).min_y;
		real max_y = blfmm_panels(leaf).max_y;
		interp_vals_bli_2d(x_basis_vals, y_basis_vals, x, y, min_x, max_x, min_y, max_y, interp_deg);
		int index = 0;
		for (int j = 0; j < interp_deg+1; j++) {
			for (int k = 0; k < interp_deg+1; k++) {
				Kokkos::atomic_add(&proxy_source_weights(leaf,index), x_basis_vals[j]*y_basis_vals[k]*charges(i));
				index += 1;
			}
		}
	}
};

struct child_to_parent_2d {
	view_panel_2d blfmm_panels;
	view_reall proxy_source_weights;
	int interp_deg;

	child_to_parent_2d(view_panel_2d& blfmm_panels_, view_reall& proxy_source_weights_, int interp_deg_) : 
					blfmm_panels(blfmm_panels_), proxy_source_weights(proxy_source_weights_), interp_deg(interp_deg_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		// interpolates proxy source weights of panel i to parent panel
		int parent = blfmm_panels(i).parent_id;
		if (parent > -1) {
			real min_x_c = blfmm_panels(i).min_x;
			real max_x_c = blfmm_panels(i).max_x;
			real min_y_c = blfmm_panels(i).min_y;
			real max_y_c = blfmm_panels(i).max_y;
			real min_x_p = blfmm_panels(parent).min_x;
			real max_x_p = blfmm_panels(parent).max_x;
			real min_y_p = blfmm_panels(parent).min_y;
			real max_y_p = blfmm_panels(parent).max_y;
			real cheb_x[max_degree+1], cheb_y[max_degree+1], x_basis_vals[max_degree+1], y_basis_vals[max_degree+1];
			bli_points_shift(cheb_x, min_x_c, max_x_c, interp_deg);
			bli_points_shift(cheb_y, min_y_c, max_y_c, interp_deg);
			int index1 = 0;
			int index2;
			for (int j1 = 0; j1 < interp_deg+1; j1++) { // child x loop
				for (int k1 = 0; k1 < interp_deg+1; k1++) { // child y loop
					interp_vals_bli_2d(x_basis_vals, y_basis_vals, cheb_x[j1], cheb_y[k1], min_x_p, max_x_p, min_y_p, max_y_p, interp_deg);
					index2 = 0;
					for (int j2 = 0; j2 < interp_deg+1; j2++) { // parent x loop
						for (int k2 = 0; k2 < interp_deg+1; k2++) { // parent y loop
							Kokkos::atomic_add(&proxy_source_weights(parent,index2), x_basis_vals[j2]*y_basis_vals[k2]*proxy_source_weights(i,index1));
							index2 += 1;
						}
					}
					index1 += 1;
				}
			}
		}
	}
};

#endif