#pragma once

#include <KokkosKernels_ArithTraits.hpp>

#include <numerix/core/scalar.hpp>

namespace numerix {

    // A numerix traits specialization does not itself register a type with Kokkos Kernels.
    template <class T>
    concept KokkosScalar = ScalarElement<T> && KokkosKernels::ArithTraits<T>::is_specialized;

    // Native axpby instantiates both simplified coefficient cases and the general expression.
    template <class X, class Y>
    concept AxpyCompatibleScalars = KokkosScalar<X> && KokkosScalar<Y> && requires(Y y, const X& x, const X& alpha, const Y& beta) {
        { y = -y };
        { y = beta * y };
        { y = x };
        { y = -x };
        { y = x + y };
        { y = x - y };
        { y = x + beta * y };
        { y = -x + y };
        { y = -x - y };
        { y = -x + beta * y };
        { y = alpha * x };
        { y = alpha * x + y };
        { y = alpha * x - y };
        { y = alpha * x + beta * y };
    };

} // namespace numerix
