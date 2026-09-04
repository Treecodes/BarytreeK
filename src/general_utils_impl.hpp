#ifndef H_GENERAL_UTIL_IMPL_H
#define H_GENERAL_UTIL_IMPL_H

#include "Kokkos_Core.hpp"

template <typename T> struct fill_2d {
	Kokkos::View<T**, Kokkos::LayoutRight> vec;
	T val;

	fill_2d(Kokkos::View<T**, Kokkos::LayoutRight>& vec_, T val_) : vec(vec_), val(val_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i, const int j) const {
		vec(i,j) = val;
	}
};

template <typename T> struct fill_1d {
	Kokkos::View<T*> vec;
	T val;

	fill_1d(Kokkos::View<T*>& vec_, T val_) : vec(vec_), val(val_) {}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int i) const {
		vec(i) = val;
	}
};

#endif