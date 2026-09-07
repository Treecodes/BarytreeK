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

struct poisson_3d_dir_sum {
	view_real xcos_t;
	view_real ycos_t;
	view_real zcos_t;
	view_real xcos_s;
	view_real ycos_s;
	view_real zcos_s;
	view_real charges;
	view_real sol;
	real eps;

	poisson_3d_dir_sum(view_real& xcos_t_, view_real& ycos_t_, view_real& zcos_t_, view_real& xcos_s_, view_real& ycos_s_, 
						view_real& zcos_s_, view_real& charges_, view_real& sol_, real eps_) : xcos_t(xcos_t_), ycos_t(ycos_t_), zcos_t(zcos_t_), 
						xcos_s(xcos_s_), ycos_s(ycos_s_), zcos_s(zcos_s_), charges(charges_), sol(sol_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		real tx = xcos_t(i);
		real ty = ycos_t(i);
		real tz = zcos_t(i);
		real sx = xcos_s(j);
		real sy = ycos_s(j);
		real sz = zcos_s(j);
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		real gfv = gfc/(Kokkos::sqrt((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps));
		Kokkos::atomic_add(&sol(i), gfv*charges(j));
	}
};

struct poisson_vel_3d_dir_sum {
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
	real eps;

	poisson_vel_3d_dir_sum(view_real& xcos_t_, view_real& ycos_t_, view_real& zcos_t_, view_real& xcos_s_, view_real& ycos_s_, 
						view_real& zcos_s_, view_real& charges_, view_real& vel_x_, view_real& vel_y_, view_real& vel_z_, real eps_) : xcos_t(xcos_t_), ycos_t(ycos_t_), zcos_t(zcos_t_), 
						xcos_s(xcos_s_), ycos_s(ycos_s_), zcos_s(zcos_s_), charges(charges_), vel_x(vel_x_), vel_y(vel_y_), vel_z(vel_z_), eps(eps_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		real tx = xcos_t(i);
		real ty = ycos_t(i);
		real tz = zcos_t(i);
		real sx = xcos_s(j);
		real sy = ycos_s(j);
		real sz = zcos_s(j);
		real gfc = -1.0/(4.0*std::numbers::pi_v<real>);
		real gfv = charges(j)*gfc/((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)+(tz-sz)*(tz-sz)+eps*eps);
		Kokkos::atomic_add(&vel_x(i), (tx-sx)*gfv);
		Kokkos::atomic_add(&vel_y(i), (ty-sy)*gfv);
		Kokkos::atomic_add(&vel_z(i), (tz-sz)*gfv);
	}
};

#endif