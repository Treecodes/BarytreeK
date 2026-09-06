#ifndef H_BLI_IMPL_H
#define H_BLI_IMPL_H

#include <Kokkos_Core.hpp>
#include <numbers>
#include "barytreek-config.h"

KOKKOS_INLINE_FUNCTION
void interp_vals_bli(real* basis_vals, real x, real min_x, real max_x, int interp_deg) {
	real x_range = 0.5*(max_x - min_x);
	real x_offset = 0.5*(max_x + min_x);
	real cheb_x[max_degree+1], bli_weights[max_degree+1], x_f_vals[max_degree+1], val;
	real eps = std::numeric_limits<real>::epsilon();
	real pi = std::numbers::pi_v<real>;

	for (int i = 0; i < interp_deg+1; i++) {
		basis_vals[i] = 0;
	}

	if (interp_deg == 0) {
		cheb_x[0] = x_offset;
		bli_weights[9] = 1;
	} else {
		for (int i = 0; i < interp_deg + 1; i++) {
			cheb_x[i] = Kokkos::cos(pi * i / interp_deg) * x_range + x_offset;
			if (i == 0) {
				bli_weights[i] = 0.5;
			} else if (i == interp_deg) {
				bli_weights[i] = 0.5 * pow(-1, i);
			} else {
				bli_weights[i] = pow(-1, i);
			}
		}
	}

	bool found_x_point = false;
	real denom_x, bliweight;
	for (int i = 0; i < interp_deg+1; i++) {
		if (Kokkos::abs(x - cheb_x[i]) < eps) {
			found_x_point = true;
			for (int j = 0; j < interp_deg+1; j++) {
				x_f_vals[j] = 0;
			}
			x_f_vals[i] = 1;
			break;
		}
	}
	if (not found_x_point) {
		denom_x = 0;
		for (int i = 0; i < interp_deg+1; i++) {
			val = bli_weights[i] / (x - cheb_x[i]);
			x_f_vals[i] = val;
			denom_x += val;
		}
		for (int i = 0; i < interp_deg+1; i++) {
			x_f_vals[i] /= denom_x;
		}
	}

	for (int i = 0; i < interp_deg+1; i++) {
		basis_vals[i] = x_f_vals[i];
	}
}

// KOKKOS_INLINE_FUNCTION
// void interp_vals_bli_2d(real* x_basis_vals, real* y_basis_vals, real x, real y, real min_x, real max_x, 
// 						real min_y, real max_y, int interp_deg) {
// 	real x_range, x_offset, y_range, y_offset, val;
// 	x_range = 0.5*(max_x - min_x);
// 	x_offset = 0.5*(max_x + min_x);
// 	y_range = 0.5*(max_y - min_y);
// 	y_offset = 0.5*(max_y + min_y);
// 	real cheb_x[max_degree+1], cheb_y[max_degree+1], bli_weights[max_degree+1];
// 	real x_f_vals[max_degree+1], y_f_vals[max_degree+1];
// 	real eps = std::numeric_limits<real>::epsilon();
// 	real pi = std::numbers::pi_v<real>;

// 	for (int i = 0; i < interp_deg+1; i++) {
// 		x_basis_vals[i] = 0;
// 		y_basis_vals[i] = 0;
// 	}

// 	if (interp_deg == 0) {
// 		cheb_x[0] = x_offset;
// 		cheb_y[0] = y_offset;
// 		bli_weights[0] = 1;
// 	} else {
// 		for (int i = 0; i < interp_deg+1; i++) {
// 			cheb_x[i] = Kokkos::cos(pi * i / interp_deg) * x_range + x_offset;
// 			cheb_y[i] = Kokkos::cos(pi * i / interp_deg) * y_range + y_offset;
// 			if (i == 0) {
// 				bli_weights[i] = 0.5;
// 			} else if (i == interp_deg) {
// 				bli_weights[i] = 0.5 * pow(-1, i);
// 			} else {
// 				bli_weights[i] = pow(-1, i);
// 			}
// 		}
// 	}

// 	bool found_x_point = false;
// 	real denom_x, bliweight;
// 	for (int i = 0; i < interp_deg+1; i++) {
// 		if (Kokkos::abs(x - cheb_x[i]) < eps) {
// 			found_x_point = true;
// 			for (int j = 0; j < interp_deg+1; j++) {
// 				x_f_vals[j] = 0;
// 			}
// 			x_f_vals[i] = 1;
// 			break;
// 		}
// 	}
// 	if (not found_x_point) {
// 		denom_x = 0;
// 		for (int i = 0; i < interp_deg+1; i++) {
// 			val = bli_weights[i] / (x - cheb_x[i]);
// 			x_f_vals[i] = val;
// 			denom_x += val;
// 		}
// 		for (int i = 0; i < interp_deg+1; i++) {
// 			x_f_vals[i] /= denom_x;
// 		}
// 	}

// 	bool found_y_point = false;
// 	real denom_y;
// 	for (int i = 0; i < interp_deg+1; i++) {
// 		if (Kokkos::abs(y - cheb_y[i]) < 1e-15) {
// 			found_y_point = true;
// 			for (int j = 0; j < interp_deg+1; j++) {
// 				y_f_vals[j] = 0;
// 			}
// 			y_f_vals[i] = 1;
// 			break;
// 		}
// 	}
// 	if (not found_y_point) {
// 		denom_y = 0;
// 		for (int i = 0; i < interp_deg+1; i++) {
// 			val = bli_weights[i] / (y - cheb_y[i]);
// 			y_f_vals[i] = val;
// 			denom_y += val;
// 		}
// 		for (int i = 0; i < interp_deg+1; i++) {
// 			y_f_vals[i] /= denom_y;
// 		}
// 	}

// 	for (int i = 0; i < interp_deg+1; i++) {
// 		x_basis_vals[i] = x_f_vals[i];
// 		y_basis_vals[i] = y_f_vals[i];
// 	}
// }

KOKKOS_INLINE_FUNCTION
void bli_points_shift(real* bli_points, real min_x, real max_x, int interp_deg) {
	real x_range = 0.5 * (max_x - min_x);
	real x_offset = 0.5 * (max_x + min_x);
	real pi = std::numbers::pi_v<real>;
	if (interp_deg == 0) {
		bli_points[0] = x_offset;
	} else {
		for (int i = 0; i <= interp_deg; i++) {
			bli_points[i] = Kokkos::cos(pi / interp_deg * i) * x_range + x_offset;
		}
	}
}

#endif