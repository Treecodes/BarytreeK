#ifndef H_DIR_SUM_IMPL_H
#define H_DIR_SUM_IMPL_H

#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "structs.hpp"

struct poisson_2d_dir_sum {
	view_real xcos_t;
	view_real ycos_t;
	view_real xcos_s;
	view_real ycos_s;
	view_real charges;
	view_real sol;
	real eps;

	poisson_2d_dir_sum(view_real& xcos_t_, view_real& ycos_t_, view_real& xcos_s_, view_real& ycos_s_, view_real& charges_, 
						view_real& sol_, real eps_) : xcos_t(xcos_t_), ycos_t(ycos_t_), xcos_s(xcos_s_), ycos_s(ycos_s_), 
						charges(charges_), sol(sol_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		real tx = xcos_t(i);
		real ty = ycos_t(i);
		real sx = xcos_s(j);
		real sy = ycos_s(j);
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		real gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
		Kokkos::atomic_add(&sol(i), gfv*charges(j));
	}
};

#endif