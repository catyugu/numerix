#pragma once

#include <concepts>
#include <cstddef>
#include <utility>

#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

#include <numerix/algebra/kokkos_scalar.hpp>
#include <numerix/core/concepts.hpp>
#include <numerix/memory/memory_space.hpp>
#include <numerix/operator/linear_operator.hpp>

namespace numerix {

    // A read-only adapter over shared Kokkos CSR storage; no implicit deep copy.
    template <ScalarElement T, std::signed_integral Ordinal = int, MemorySpace Mem = DefaultMemorySpace, std::integral Offset = std::size_t>
        requires(!std::same_as<Offset, bool>)
    class SparseOperator {
    public:
        using value_type = T;
        using memory_space = Mem;
        using size_type = std::size_t;
        using matrix_type = KokkosSparse::CrsMatrix<T, Ordinal, Kokkos::Device<typename Mem::execution_space, Mem>, void, Offset>;

        using const_matrix_type = typename matrix_type::const_type;

        explicit SparseOperator(const_matrix_type matrix) : matrix_(std::move(matrix)) { }

        size_type Rows() const noexcept { return static_cast<size_type>(matrix_.numRows()); }
        size_type Columns() const noexcept { return static_cast<size_type>(matrix_.numCols()); }

        template <class Exec, class X, class Y>
            requires(AccessibleFrom<Exec, Mem> && KokkosScalar<T> && ((DenseVector<X, Exec> && MutableDenseVector<Y, Exec>) || (DenseMultiVector<X, Exec> && MutableDenseMultiVector<Y, Exec>))
                && std::same_as<ScalarOf<X>, T> && std::same_as<ScalarOf<Y>, T>)
        void Apply(const Exec& exec, const X& x, const Y& y) const
        {
            Apply(exec, T(1), x, T(0), y);
        }

        template <class Exec, class X, class Y>
            requires(AccessibleFrom<Exec, Mem> && KokkosScalar<T> && ((DenseVector<X, Exec> && MutableDenseVector<Y, Exec>) || (DenseMultiVector<X, Exec> && MutableDenseMultiVector<Y, Exec>))
                && std::same_as<ScalarOf<X>, T> && std::same_as<ScalarOf<Y>, T>)
        void Apply(const Exec& exec, T alpha, const X& x, T beta, const Y& y) const
        {
            // Keep TPL setup and mutable resources out of the core operator's hot path.
            KokkosSparse::SPMVHandle<Exec, const_matrix_type, X, Y> handle(KokkosSparse::SPMV_NATIVE);
            KokkosSparse::spmv(exec, &handle, "N", alpha, matrix_, x, beta, y);
        }

    private:
        const_matrix_type matrix_;
    };

} // namespace numerix
