#pragma once

#include <concepts>

#include <KokkosKernels_ArithTraits.hpp>

#include <numerix/core/scalar.hpp>

namespace numerix {

    // A numerix traits specialization does not itself register a type with Kokkos Kernels.
    template <class T>
    concept KokkosScalar = ScalarElement<T> && KokkosKernels::ArithTraits<T>::is_specialized;

    // Conversion is the interface policy; kernel arithmetic is checked by Kokkos Kernels.
    template <class X, class Y>
    concept AxpyCompatibleScalars = KokkosScalar<X> && KokkosScalar<Y> && std::convertible_to<X, Y>;

} // namespace numerix
