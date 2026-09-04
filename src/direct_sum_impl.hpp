#ifndef H_DIR_SUM_IMPL_H
#define H_DIR_SUM_IMPL_H

#include <Kokkos_Core.hpp>

#include "barytreek-config.h"
#include "structs.hpp"

struct poisson_2d_dir_sum {
	view_real xcos;
	view_real ycos;
	view_real charges;
	view_real sol;
	real eps;

	poisson_2d_dir_sum(view_real& xcos_, view_real& ycos_, view_real& charges_, view_real& sol_, real eps_) :
						xcos(xcos_), ycos(ycos_), charges(charges_), sol(sol_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		real tx = xcos(i);
		real ty = ycos(i);
		real sx = xcos(j);
		real sy = ycos(j);
		real gfc = 1.0/(4.0*std::numbers::pi_v<real>);
		real gfv = gfc*Kokkos::log((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+eps*eps);
		Kokkos::atomic_add(&sol(i), gfv*charges(j));
	}
};

#endif