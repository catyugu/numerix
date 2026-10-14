#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include <numerix/numerix.hpp>

#include "test_support.hpp"

namespace constraint_test {
    struct MissingOperations { };
    struct WrongConjugate { };
    struct WrongMagnitude { };
    struct TraitValue {
        double value;
    };
}

namespace numerix {
    template <>
    struct ScalarTraits<constraint_test::MissingOperations> {
        using real_type = double;
    };
    template <>
    struct ScalarTraits<constraint_test::WrongConjugate> {
        using real_type = double;
        static int Conj(constraint_test::WrongConjugate) noexcept { return 0; }
        static double SquaredMagnitude(constraint_test::WrongConjugate) noexcept { return 0; }
    };
    template <>
    struct ScalarTraits<constraint_test::WrongMagnitude> {
        using real_type = double;
        static constraint_test::WrongMagnitude Conj(constraint_test::WrongMagnitude x) noexcept { return x; }
        static void SquaredMagnitude(constraint_test::WrongMagnitude) noexcept { }
    };
    // Traits may return references; helpers must materialize values. No unused alias is required.
    template <>
    struct ScalarTraits<constraint_test::TraitValue> {
        using real_type = double;
        static constexpr const constraint_test::TraitValue& Conj(const constraint_test::TraitValue& x) noexcept { return x; }
        static constexpr double SquaredMagnitude(const constraint_test::TraitValue& x) noexcept { return x.value * x.value; }
    };

    namespace {
        using E = DefaultExecutionSpace;
        using Z = Kokkos::complex<double>;
        template <class T>
        using View = Kokkos::View<T*, DefaultMemorySpace>;

        template <class T>
        concept CanStore = requires {
            typename Vector<T>;
            typename MultiVector<T>;
            typename StaticVector<T, 2>;
            typename StaticMatrix<T, 2, 2>;
        };
        template <class X>
        concept CanScale = requires(const E& e, const X& x) { Scale(e, ScalarOf<X>(1), x); };
        template <class X, class Y = X, class A = ScalarOf<X>>
        concept CanAxpy = requires(const E& e, const X& x, const Y& y) { Axpy(e, A(1), x, y); };
        template <class X>
        concept CanDot = requires(const E& e, const X& x) { Dot(e, x, x); };
        template <class X>
        concept CanNorm = requires(const E& e, const X& x) { Norm2(e, x); };
        template <class X, class R>
        concept CanColumnDots = requires(const E& e, const X& x, const R& r) { ColumnDots(e, x, x, r); };

        static_assert(!Scalar<constraint_test::MissingOperations>);
        static_assert(!Scalar<constraint_test::WrongConjugate>);
        static_assert(!Scalar<constraint_test::WrongMagnitude>);
        static_assert(Scalar<constraint_test::TraitValue>);
        static_assert(Scalar<const double&>);
        static_assert(CanStore<double> && CanStore<Z> && CanStore<constraint_test::TraitValue>);
        static_assert(!CanStore<double&> && !CanStore<const double>);
        static_assert(std::same_as<decltype(Conj<double&>(std::declval<double&>())), double>);
        static_assert(!CanScale<View<int>> && !CanAxpy<View<int>> && !CanDot<View<int>> && !CanNorm<View<int>>);
        static_assert(!CanScale<View<const double>> && !CanAxpy<View<double>, View<const double>>);
        static_assert(CanScale<View<double>> && CanDot<View<Z>> && CanNorm<View<Z>>);
        static_assert(CanAxpy<View<float>, View<double>>);
        static_assert(CanAxpy<View<double>, View<Z>>);
        static_assert(!CanAxpy<View<double>, View<Z>, Z>);
        static_assert(!CanAxpy<View<Z>, View<double>>);
        using WideComplex = Kokkos::complex<long double>;
        static_assert(Scalar<WideComplex>);
        static_assert(!CanScale<View<WideComplex>> && !CanDot<View<WideComplex>>);
        static_assert(!CanDot<View<constraint_test::TraitValue>>);
        static_assert(!CanColumnDots<Kokkos::View<WideComplex**, DefaultMemorySpace>, View<WideComplex>>);
        static_assert(!ScaledLinearOperatorFor<SparseOperator<WideComplex>, E, View<WideComplex>>);

        TEST(ConstraintTest, TraitReferencesProduceIndependentValues)
        {
            constexpr constraint_test::TraitValue original {3.0};
            constexpr auto conjugated = Conj(original);
            static_assert(conjugated.value == 3.0);
            static_assert(SquaredMagnitude(original) == 9.0);
            static_assert(!std::is_reference_v<decltype(conjugated)>);
            double x = 4.0;
            const auto copied = Conj<double&>(x);
            x = 0.0;
            EXPECT_DOUBLE_EQ(copied, 4.0);
        }

        TEST(ConstraintTest, CustomScalarStorageDoesNotRequireBackendRegistration)
        {
            const auto exec = test_support::Exec();
            Vector<constraint_test::TraitValue> x(2);
            Kokkos::deep_copy(exec, x.View(), constraint_test::TraitValue {3.0});
            auto y = Clone(exec, x);
            Kokkos::deep_copy(exec, x.View(), constraint_test::TraitValue {0.0});
            exec.fence();
            const auto values = test_support::ToHost(y);
            EXPECT_DOUBLE_EQ(values[0].value, 3.0);
            EXPECT_DOUBLE_EQ(values[1].value, 3.0);
        }

        TEST(ConstraintTest, StaticVectorAcceptsConvertibleCoefficients)
        {
            constexpr StaticVector<double, 2> real(Kokkos::Array<double, 2> {1.0, 2.0});
            constexpr auto left = 2 * real;
            constexpr auto right = real * 2;
            static_assert(left[0] == 2.0 && right[1] == 4.0);
            constexpr StaticVector<Z, 2> complex(Kokkos::Array<Z, 2> {Z(1.0, 2.0), Z(3.0, -1.0)});
            constexpr auto scaled = 2.0 * complex;
            static_assert(scaled[0].real() == 2.0 && scaled[0].imag() == 4.0);
            EXPECT_EQ((complex * 2.0)[1], Z(6.0, -2.0));
        }

        TEST(ConstraintTest, MixedPrecisionAxpyPreservesOutputTypeAndValue)
        {
            const auto exec = test_support::Exec();
            View<float> x("x", 3);
            Vector<double> y(3);
            Kokkos::deep_copy(exec, x, 2.0f);
            Kokkos::deep_copy(exec, y.View(), 3.0);
            Axpy(exec, 1.5f, x, y.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(y), std::vector<double>({6.0, 6.0, 6.0}));
            Vector<Z> z(3);
            Kokkos::deep_copy(exec, z.View(), Z(1.0, 1.0));
            Axpy(exec, 2.0, y.View(), z.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(z), std::vector<Z>({Z(13.0, 1.0), Z(13.0, 1.0), Z(13.0, 1.0)}));
        }

        TEST(ConstraintTest, SparseOperatorAcceptsReadOnlyCsr)
        {
            const auto exec = test_support::Exec();
            using Matrix = SparseOperator<double>::matrix_type;
            Matrix::row_map_type::non_const_type rows("rows", 2);
            Matrix::index_type columns("columns", 1);
            Matrix::values_type entries("entries", 1);
            auto host_rows = Kokkos::create_mirror_view(rows);
            host_rows(0) = 0;
            host_rows(1) = 1;
            Kokkos::deep_copy(exec, rows, host_rows);
            Kokkos::deep_copy(exec, columns, 0);
            Kokkos::deep_copy(exec, entries, 4.0);
            const Matrix::const_type matrix(Matrix("diagonal", 1, 1, 1, entries, rows, columns));
            const SparseOperator<double> op(matrix);
            Vector<double> x(1), y(1);
            Kokkos::deep_copy(exec, x.View(), 3.0);
            op.Apply(exec, x.View(), y.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(y), std::vector<double>({12.0}));
        }
    }
}
