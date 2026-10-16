#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <numerix/algebra/multivector.hpp>
#include <numerix/operator/sparse_operator.hpp>

#include "test_support.hpp"

namespace numerix {
    namespace {
        using test_support::Exec;
        using Complex = Kokkos::complex<double>;
        constexpr double kTolerance = 1e-12;

        template <Scalar T, std::integral Offset = std::size_t>
        auto MakeMatrix(std::initializer_list<T> values)
        {
            using Matrix = typename SparseOperator<T, int, DefaultMemorySpace, Offset>::matrix_type;
            typename Matrix::row_map_type::non_const_type rows("rows", 4);
            typename Matrix::index_type columns("columns", 4);
            typename Matrix::values_type entries("values", 4);
            auto hr = Kokkos::create_mirror_view(rows);
            auto hc = Kokkos::create_mirror_view(columns);
            auto hv = Kokkos::create_mirror_view(entries);
            // Rectangular 3 x 4 matrix with an empty middle row.
            hr(0) = 0;
            hr(1) = 2;
            hr(2) = 2;
            hr(3) = 4;
            hc(0) = 0;
            hc(1) = 3;
            hc(2) = 1;
            hc(3) = 2;
            std::size_t i = 0;
            for (const auto& value : values) {
                hv(i++) = value;
            }
            Kokkos::deep_copy(rows, hr);
            Kokkos::deep_copy(columns, hc);
            Kokkos::deep_copy(entries, hv);
            return Matrix("test::matrix", 3, 4, 4, entries, rows, columns);
        }

        using View = Vector<double>::view_type;
        static_assert(ScaledLinearOperatorFor<SparseOperator<double>, DefaultExecutionSpace, View>);
        static_assert(!LinearOperatorFor<SparseOperator<double>, DefaultExecutionSpace, View, Vector<double>::const_view_type>);
        static_assert(!LinearOperatorFor<SparseOperator<double>, DefaultExecutionSpace, View, Vector<Complex>::view_type>);
        static_assert(std::is_same_v<SparseOperator<double, long long, DefaultMemorySpace, std::size_t>::matrix_type::ordinal_type, long long>);

        TEST(SparseOperatorTest, InvalidVectorExtentsUseBackendDiagnostics)
        {
            const auto exec = Exec();
            const SparseOperator<double> op(MakeMatrix<double>({2.0, -1.0, 3.0, 4.0}));
            Vector<double> x(4), y(3), short_x(3), short_y(2), long_x(5), long_y(4);
            EXPECT_THROW(op.Apply(exec, short_x.View(), y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, x.View(), short_y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, long_x.View(), y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, x.View(), long_y.View()), std::runtime_error);
        }

        TEST(SparseOperatorTest, InvalidMultiVectorExtentsUseBackendDiagnostics)
        {
            const auto exec = Exec();
            const SparseOperator<double> op(MakeMatrix<double>({2.0, -1.0, 3.0, 4.0}));
            MultiVector<double> x(4, 2), y(3, 2), short_x(3, 2), short_y(2, 2), long_x(5, 2), long_y(4, 2), extra_column(3, 3);
            EXPECT_THROW(op.Apply(exec, short_x.View(), y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, x.View(), short_y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, long_x.View(), y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, x.View(), long_y.View()), std::runtime_error);
            EXPECT_THROW(op.Apply(exec, x.View(), extra_column.View()), std::runtime_error);
        }

        TEST(SparseOperatorTest, NativeSignedOffsetsAreSupportedWithoutConversion)
        {
            const auto exec = Exec();
            using Op = SparseOperator<double, int, DefaultMemorySpace, int>;
            const Op op(MakeMatrix<double, int>({2.0, -1.0, 3.0, 4.0}));
            Vector<double> x(4), y(3);
            Kokkos::deep_copy(exec, x.View(), 2.0);
            op.Apply(exec, x.View(), y.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(y), std::vector<double>({2.0, 0.0, 14.0}));
        }

        TEST(SparseOperatorTest, RectangularMultiplyMatchesRowsIncludingEmptyRow)
        {
            const auto exec = Exec();
            const SparseOperator<double> op(MakeMatrix<double>({2.0, -1.0, 3.0, 4.0}));
            Vector<double> x(4), y(3);
            test_support::FillRamp(x, 1.0, 1.0);
            Kokkos::deep_copy(exec, y.View(), std::numeric_limits<double>::quiet_NaN());
            op.Apply(exec, std::as_const(x).View(), y.View());
            exec.fence();
            EXPECT_EQ(op.Rows(), 3u);
            EXPECT_EQ(op.Columns(), 4u);
            EXPECT_EQ(test_support::ToHost(y), std::vector<double>({-2.0, 0.0, 18.0}));
        }

        TEST(SparseOperatorTest, LinearityAndFusedApplyMatchSeparateOperations)
        {
            const auto exec = Exec();
            const SparseOperator<double> op(MakeMatrix<double>({2.0, -1.0, 3.0, 4.0}));
            Vector<double> x(4), z(4), ax(3), az(3), fused(3);
            test_support::FillRamp(x, 1.0, 1.0);
            test_support::FillRamp(z, 2.0, -0.5);
            op.Apply(exec, x.View(), ax.View());
            op.Apply(exec, z.View(), az.View());
            auto combined = Clone(exec, x);
            Scale(exec, 2.0, combined.View());
            Axpy(exec, -3.0, z.View(), combined.View());
            op.Apply(exec, combined.View(), fused.View());
            Scale(exec, 2.0, ax.View());
            Axpy(exec, -3.0, az.View(), ax.View());
            auto reference = test_support::ToHost(ax);
            auto actual = test_support::ToHost(fused);
            for (std::size_t row = 0; row < 3; ++row) {
                EXPECT_NEAR(actual[row], reference[row], kTolerance);
            }
            Kokkos::deep_copy(exec, fused.View(), 5.0);
            op.Apply(exec, 2.0, x.View(), -3.0, fused.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(fused), std::vector<double>({-19.0, -15.0, 21.0}));
        }

        TEST(SparseOperatorTest, ComplexMultiplyDoesNotConjugateMatrix)
        {
            const auto exec = Exec();
            const SparseOperator<Complex> op(MakeMatrix<Complex>({{2.0, 1.0}, {-1.0, 2.0}, {3.0, -1.0}, {4.0, 1.0}}));
            Vector<Complex> x(4), y(3);
            Kokkos::deep_copy(exec, x.View(), Complex(1.0, 2.0));
            Kokkos::deep_copy(exec, y.View(), Complex(2.0, -1.0));
            op.Apply(exec, Complex(0.0, 1.0), x.View(), Complex(1.0, -1.0), y.View());
            exec.fence();
            const auto actual = test_support::ToHost(y);
            const Complex expected[] = {Complex(-4.0, -8.0), Complex(1.0, -3.0), Complex(-13.0, 4.0)};
            for (std::size_t row = 0; row < 3; ++row) {
                EXPECT_NEAR(actual[row].real(), expected[row].real(), kTolerance);
                EXPECT_NEAR(actual[row].imag(), expected[row].imag(), kTolerance);
            }
        }

        TEST(SparseOperatorTest, HermitianPositiveDefiniteMatrixRespectsInnerProduct)
        {
            const auto exec = Exec();
            using Matrix = SparseOperator<Complex>::matrix_type;
            Matrix::row_map_type::non_const_type rows("rows", 3);
            Matrix::index_type columns("columns", 4);
            Matrix::values_type entries("entries", 4);
            auto hr = Kokkos::create_mirror_view(rows);
            auto hc = Kokkos::create_mirror_view(columns);
            auto hv = Kokkos::create_mirror_view(entries);
            hr(0) = 0;
            hr(1) = 2;
            hr(2) = 4;
            hc(0) = 0;
            hc(1) = 1;
            hc(2) = 0;
            hc(3) = 1;
            hv(0) = Complex(3, 0);
            hv(1) = Complex(0, 1);
            hv(2) = Complex(0, -1);
            hv(3) = Complex(2, 0);
            Kokkos::deep_copy(rows, hr);
            Kokkos::deep_copy(columns, hc);
            Kokkos::deep_copy(entries, hv);
            const SparseOperator<Complex> op(Matrix("hermitian", 2, 2, 4, entries, rows, columns));
            Vector<Complex> x(2), y(2), ax(2), ay(2);
            auto hx = Kokkos::create_mirror_view(x.View());
            auto hy = Kokkos::create_mirror_view(y.View());
            hx(0) = Complex(1, 2);
            hx(1) = Complex(-2, 1);
            hy(0) = Complex(3, -1);
            hy(1) = Complex(1, 4);
            Kokkos::deep_copy(exec, x.View(), hx);
            Kokkos::deep_copy(exec, y.View(), hy);
            op.Apply(exec, x.View(), ax.View());
            op.Apply(exec, y.View(), ay.View());
            const auto lhs = Dot(exec, x.View(), ay.View());
            const auto rhs = Dot(exec, ax.View(), y.View());
            EXPECT_NEAR(lhs.real(), rhs.real(), kTolerance);
            EXPECT_NEAR(lhs.imag(), rhs.imag(), kTolerance);
            const auto quadratic = Dot(exec, x.View(), ax.View());
            EXPECT_GT(quadratic.real(), 0.0);
            EXPECT_NEAR(quadratic.imag(), 0.0, kTolerance);
        }

        TEST(SparseOperatorTest, ZeroMatrixClearsOutputWithBetaZero)
        {
            const auto exec = Exec();
            using Matrix = SparseOperator<double>::matrix_type;
            Matrix::row_map_type::non_const_type rows("rows", 4);
            Matrix::index_type columns("columns", 0);
            Matrix::values_type entries("entries", 0);
            Kokkos::deep_copy(exec, rows, 0);
            const SparseOperator<double> op(Matrix("zero", 3, 4, 0, entries, rows, columns));
            Vector<double> x(4), y(3);
            Kokkos::deep_copy(exec, x.View(), 2.0);
            Kokkos::deep_copy(exec, y.View(), std::numeric_limits<double>::quiet_NaN());
            op.Apply(exec, x.View(), y.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(y), std::vector<double>({0.0, 0.0, 0.0}));
        }

        TEST(SparseOperatorTest, MultiVectorMultiplyMatchesIndependentColumns)
        {
            const auto exec = Exec();
            const SparseOperator<double> op(MakeMatrix<double>({2.0, -1.0, 3.0, 4.0}));
            MultiVector<double> x(4, 2), y(3, 2);
            Kokkos::deep_copy(exec, x.View(), 2.0);
            Kokkos::deep_copy(exec, y.View(), 1.0);
            op.Apply(exec, 3.0, std::as_const(x).View(), -2.0, y.View());
            exec.fence();
            const auto actual = Kokkos::create_mirror_view_and_copy(DefaultHostMemorySpace {}, y.View());
            for (std::size_t col = 0; col < 2; ++col) {
                Vector<double> reference(3);
                Kokkos::deep_copy(exec, reference.View(), 1.0);
                op.Apply(exec, 3.0, Kokkos::subview(x.View(), Kokkos::ALL, col), -2.0, reference.View());
                exec.fence();
                const auto expected = test_support::ToHost(reference);
                for (std::size_t row = 0; row < 3; ++row) {
                    EXPECT_NEAR(actual(row, col), expected[row], kTolerance);
                }
            }
        }
    }
}
