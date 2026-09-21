#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <numerix/numerix.hpp>

#include "test_support.hpp"

namespace numerix {
    namespace {

        using test_support::Exec;
        using test_support::RangeOf;
        using test_support::ToHost;

        using View = Kokkos::View<double*, DefaultMemorySpace>;

        // ownership/value semantics 必须在编译期就被钉死：Vector 是 owning、move-only 存储。
        static_assert(!std::is_copy_constructible_v<Vector<double>>);
        static_assert(!std::is_copy_assignable_v<Vector<double>>);
        static_assert(std::is_move_constructible_v<Vector<double>>);
        static_assert(std::is_move_assignable_v<Vector<double>>);

        // const Vector 只能交出 const 数据，与 Kokkos 的 View<double*> / View<const double*> 一致。
        static_assert(std::is_same_v<decltype(std::declval<const Vector<double>&>().Data()), const double*>);
        static_assert(std::is_same_v<decltype(std::declval<Vector<double>&>().Data()), double*>);
        static_assert(std::is_same_v<decltype(std::declval<const Vector<double>&>().View()), Vector<double>::const_view_type>);
        static_assert(std::is_same_v<decltype(std::declval<Vector<double>&>().View()), Vector<double>::view_type>);

        TEST(VectorTest, AllocatesRequestedSize)
        {
            const Vector<double> v(8);
            EXPECT_EQ(v.Size(), 8u);
            EXPECT_NE(v.Data(), nullptr);
        }

        TEST(VectorTest, MoveTransfersOwnershipWithoutCopy)
        {
            Vector<double> source(4);
            Kokkos::deep_copy(source.View(), 3.0);
            const double* const raw = source.Data();

            const Vector<double> moved(std::move(source));

            EXPECT_EQ(moved.Size(), 4u);
            EXPECT_EQ(moved.Data(), raw);
            EXPECT_DOUBLE_EQ(Dot(Exec(), moved.View(), moved.View()), 36.0);
        }

        TEST(VectorTest, AdoptsExistingViewWithoutCopy)
        {
            View storage("test::storage", 3);
            Kokkos::deep_copy(storage, 7.0);
            const double* const raw = storage.data();

            const Vector<double> wrapped {std::move(storage)};

            EXPECT_EQ(wrapped.Data(), raw);
            EXPECT_DOUBLE_EQ(Dot(Exec(), wrapped.View(), wrapped.View()), 147.0);
        }

        TEST(VectorTest, CloneDoesNotShareStorage)
        {
            Vector<double> source(3);
            Kokkos::deep_copy(source.View(), 4.0);

            Vector<double> copy = Clone(Exec(), source);
            Scale(Exec(), 0.0, source.View());

            EXPECT_NE(copy.Data(), source.Data());
            EXPECT_EQ(ToHost(copy), std::vector<double>({4.0, 4.0, 4.0}));
            EXPECT_EQ(ToHost(source), std::vector<double>({0.0, 0.0, 0.0}));
        }

        TEST(VectorTest, DeepCopyOverwritesTarget)
        {
            Vector<double> source(2);
            Vector<double> target(2);
            Kokkos::deep_copy(source.View(), 5.0);
            Kokkos::deep_copy(target.View(), 0.0);

            DeepCopy(Exec(), target, source);
            Scale(Exec(), 0.0, source.View());

            EXPECT_EQ(ToHost(target), std::vector<double>({5.0, 5.0}));
        }

        TEST(VectorTest, DotAndNorm2AgreeWithScalarMath)
        {
            Vector<double> x(4);
            Kokkos::deep_copy(x.View(), 2.0);

            EXPECT_DOUBLE_EQ(Dot(Exec(), x.View(), x.View()), 16.0);
            EXPECT_DOUBLE_EQ(Norm2(Exec(), x.View()), 4.0);

            // 不带 execution-space instance 的便利重载给出同样的结果。
            EXPECT_DOUBLE_EQ(Dot(x.View(), x.View()), 16.0);
            EXPECT_DOUBLE_EQ(Norm2(x.View()), 4.0);
        }

        TEST(VectorTest, AxpyAndScale)
        {
            Vector<double> x(3);
            Vector<double> y(3);
            Kokkos::deep_copy(x.View(), 1.0);
            Kokkos::deep_copy(y.View(), 3.0);

            Axpy(Exec(), 2.0, x.View(), y.View());
            Scale(Exec(), 0.5, y.View());

            EXPECT_EQ(ToHost(y), std::vector<double>({2.5, 2.5, 2.5}));
        }

        // 复数内积只对第二个参数线性：Dot(x, y) = conj(x) . y。
        TEST(VectorTest, ComplexDotConjugatesFirstArgument)
        {
            Vector<Kokkos::complex<double>> x(2);
            Vector<Kokkos::complex<double>> y(2);

            auto x_host = Kokkos::create_mirror_view(x.View());
            x_host(0) = Kokkos::complex<double>(1.0, 2.0);
            x_host(1) = Kokkos::complex<double>(3.0, -1.0);
            Kokkos::deep_copy(x.View(), x_host);

            auto y_host = Kokkos::create_mirror_view(y.View());
            y_host(0) = Kokkos::complex<double>(0.0, 0.0);
            y_host(1) = Kokkos::complex<double>(1.0, 0.0);
            Kokkos::deep_copy(y.View(), y_host);

            const Kokkos::complex<double> self_dot = Dot(Exec(), x.View(), x.View());
            EXPECT_DOUBLE_EQ(self_dot.real(), 15.0);
            EXPECT_DOUBLE_EQ(self_dot.imag(), 0.0);
            EXPECT_DOUBLE_EQ(Norm2(Exec(), x.View()), std::sqrt(15.0));

            const Kokkos::complex<double> xy = Dot(Exec(), x.View(), y.View());
            EXPECT_DOUBLE_EQ(xy.real(), 3.0);
            EXPECT_DOUBLE_EQ(xy.imag(), 1.0);

            const Kokkos::complex<double> yx = Dot(Exec(), y.View(), x.View());
            EXPECT_DOUBLE_EQ(yx.real(), 3.0);
            EXPECT_DOUBLE_EQ(yx.imag(), -1.0);
        }

        TEST(VectorTest, ComplexScaleUsesKokkosComplexScalar)
        {
            Vector<Kokkos::complex<double>> x(2);
            auto x_host = Kokkos::create_mirror_view(x.View());
            x_host(0) = Kokkos::complex<double>(1.0, 2.0);
            x_host(1) = Kokkos::complex<double>(3.0, -1.0);
            Kokkos::deep_copy(x.View(), x_host);

            Scale(Exec(), Kokkos::complex<double>(0.0, 1.0), x.View());

            const auto result = ToHost(x);
            EXPECT_DOUBLE_EQ(result[0].real(), -2.0);
            EXPECT_DOUBLE_EQ(result[0].imag(), 1.0);
            EXPECT_DOUBLE_EQ(result[1].real(), 1.0);
            EXPECT_DOUBLE_EQ(result[1].imag(), 3.0);
        }

        // 填充入口必须按 View 的 index_type 取索引：这里用一个大到 int 放不下的 extent 做回归。
        TEST(VectorTest, RangePolicyFollowsViewIndexType)
        {
            constexpr std::size_t kBig = 1u << 20;
            Vector<double> v(kBig);

            Kokkos::deep_copy(v.View(), 1.0);
            const auto values = v.View();
            Kokkos::parallel_for(
                "test::IndexTypeProbe", RangeOf(Exec(), values), KOKKOS_LAMBDA(test_support::index_type i) { values(i) += 1.0; });

            EXPECT_DOUBLE_EQ(Dot(Exec(), v.View(), v.View()), static_cast<double>(kBig) * 4.0);
        }

    } // namespace
} // namespace numerix
