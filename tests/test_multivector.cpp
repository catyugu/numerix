#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include <numerix/algebra/blas1.hpp>
#include <numerix/algebra/multivector.hpp>

#include "test_support.hpp"

namespace numerix {
    namespace {
        using test_support::Exec;
        using Complex = Kokkos::complex<double>;
        constexpr double kTolerance = 1e-12;

        static_assert(!std::is_copy_constructible_v<MultiVector<double>>);
        static_assert(!std::is_copy_assignable_v<MultiVector<double>>);
        static_assert(std::is_nothrow_move_constructible_v<MultiVector<double>>);
        static_assert(std::is_nothrow_move_assignable_v<MultiVector<double>>);
        static_assert(DenseMultiVector<MultiVector<double>::const_view_type>);
        static_assert(!MutableDenseMultiVector<MultiVector<double>::const_view_type>);
        static_assert(!DenseMultiVector<Vector<double>::view_type>);
        static_assert(std::is_same_v<decltype(std::declval<const MultiVector<double>&>().View()), MultiVector<double>::const_view_type>);
        static_assert(std::is_same_v<MultiVector<double>::view_type::array_layout, Kokkos::LayoutLeft>);

        TEST(MultiVectorTest, MoveAndAdoptPreserveAllocation)
        {
            MultiVector<double>::view_type storage("test::block", 5, 3);
            const auto* allocation = storage.data();
            MultiVector<double> original(std::move(storage));
            MultiVector<double> moved(std::move(original));
            EXPECT_EQ(moved.Rows(), 5u);
            EXPECT_EQ(moved.Columns(), 3u);
            EXPECT_EQ(moved.View().data(), allocation);
            MultiVector<double> target(1, 1);
            target = std::move(moved);
            EXPECT_EQ(target.View().data(), allocation);
        }

        TEST(MultiVectorTest, CloneAndDeepCopyAreIndependent)
        {
            const auto exec = Exec();
            MultiVector<double> source(4, 2);
            Kokkos::deep_copy(exec, source.View(), 3.0);
            auto cloned = Clone(exec, source);
            MultiVector<double> copied(4, 2);
            DeepCopy(exec, copied, source);
            Kokkos::deep_copy(exec, source.View(), 0.0);
            exec.fence();
            EXPECT_NE(source.View().data(), cloned.View().data());
            for (std::size_t col = 0; col < source.Columns(); ++col) {
                EXPECT_DOUBLE_EQ(Norm2(exec, Kokkos::subview(cloned.View(), Kokkos::ALL, col)), 6.0);
                EXPECT_DOUBLE_EQ(Norm2(exec, Kokkos::subview(copied.View(), Kokkos::ALL, col)), 6.0);
            }
        }

        TEST(MultiVectorTest, ColumnViewsSupportExistingBlas)
        {
            const auto exec = Exec();
            MultiVector<double> block(4, 2);
            Kokkos::deep_copy(exec, block.View(), 1.0);
            auto first = Kokkos::subview(block.View(), Kokkos::ALL, 0);
            auto second = Kokkos::subview(block.View(), Kokkos::ALL, 1);
            Scale(exec, 2.0, first);
            Axpy(exec, 3.0, first, second);
            EXPECT_DOUBLE_EQ(Dot(exec, first, second), 56.0);
            EXPECT_DOUBLE_EQ(Norm2(exec, second), 14.0);
        }

        TEST(MultiVectorTest, ColumnDotsMatchIndependentVectorDots)
        {
            const auto exec = Exec();
            MultiVector<double> x(4, 3);
            MultiVector<double> y(4, 3);
            auto hx = Kokkos::create_mirror_view(x.View());
            auto hy = Kokkos::create_mirror_view(y.View());
            for (std::size_t col = 0; col < 3; ++col) {
                for (std::size_t row = 0; row < 4; ++row) {
                    hx(row, col) = static_cast<double>(row + col + 1);
                    hy(row, col) = static_cast<double>(2 * row + col);
                }
            }
            Kokkos::deep_copy(exec, x.View(), hx);
            Kokkos::deep_copy(exec, y.View(), hy);
            Vector<double> dots(3);
            ColumnDots(exec, std::as_const(x).View(), std::as_const(y).View(), dots.View());
            exec.fence();
            const auto result = Kokkos::create_mirror_view_and_copy(DefaultHostMemorySpace {}, dots.View());
            for (std::size_t col = 0; col < 3; ++col) {
                Vector<double> vx(4);
                Vector<double> vy(4);
                Kokkos::deep_copy(exec, vx.View(), Kokkos::subview(x.View(), Kokkos::ALL, col));
                Kokkos::deep_copy(exec, vy.View(), Kokkos::subview(y.View(), Kokkos::ALL, col));
                EXPECT_NEAR(result(col), Dot(exec, vx.View(), vy.View()), kTolerance);
            }
        }

        TEST(MultiVectorTest, ComplexColumnDotsAreConjugateSymmetricAndPositive)
        {
            const auto exec = Exec();
            // LayoutRight columns are strided: the API must not assume contiguous columns.
            Kokkos::View<Complex**, Kokkos::LayoutRight, DefaultMemorySpace> x("x", 3, 2);
            decltype(x) y("y", 3, 2);
            auto hx = Kokkos::create_mirror_view(x);
            auto hy = Kokkos::create_mirror_view(y);
            for (std::size_t col = 0; col < 2; ++col) {
                for (std::size_t row = 0; row < 3; ++row) {
                    hx(row, col) = Complex(static_cast<double>(row + 1), static_cast<double>(col + 2));
                    hy(row, col) = Complex(static_cast<double>(col + 1), -static_cast<double>(row));
                }
            }
            Kokkos::deep_copy(exec, x, hx);
            Kokkos::deep_copy(exec, y, hy);
            Vector<Complex> xy(2), yx(2), xx(2);
            ColumnDots(exec, x, y, xy.View());
            ColumnDots(exec, y, x, yx.View());
            ColumnDots(exec, x, x, xx.View());
            exec.fence();
            const auto hxy = test_support::ToHost(xy);
            const auto hyx = test_support::ToHost(yx);
            const auto hxx = test_support::ToHost(xx);
            for (std::size_t col = 0; col < 2; ++col) {
                Complex expected {};
                for (std::size_t row = 0; row < 3; ++row) {
                    expected += Conj(hx(row, col)) * hy(row, col);
                }
                EXPECT_NEAR(hxy[col].real(), expected.real(), kTolerance);
                EXPECT_NEAR(hxy[col].imag(), expected.imag(), kTolerance);
                EXPECT_NEAR(hxy[col].real(), hyx[col].real(), kTolerance);
                EXPECT_NEAR(hxy[col].imag(), -hyx[col].imag(), kTolerance);
                EXPECT_GT(hxx[col].real(), 0.0);
                EXPECT_NEAR(hxx[col].imag(), 0.0, kTolerance);
            }
        }

        TEST(MultiVectorTest, EmptyShapesRetainDimensionsAndZeroInnerProducts)
        {
            const auto exec = Exec();
            MultiVector<double> empty_rows(0, 3);
            Vector<double> dots(3);
            Kokkos::deep_copy(exec, dots.View(), 42.0);
            ColumnDots(exec, empty_rows.View(), empty_rows.View(), dots.View());
            exec.fence();
            EXPECT_EQ(test_support::ToHost(dots), std::vector<double>({0.0, 0.0, 0.0}));
            MultiVector<double> empty_columns(4, 0);
            Vector<double> no_dots(0);
            ColumnDots(exec, empty_columns.View(), empty_columns.View(), no_dots.View());
            exec.fence();
            EXPECT_EQ(Clone(exec, empty_columns).Rows(), 4u);
            EXPECT_EQ(Clone(exec, empty_rows).Columns(), 3u);
        }
    }
}
