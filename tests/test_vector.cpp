#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include <numerix/numerix.hpp>

namespace numerix {
    namespace {

        // 读回主机内存：不假设默认内存空间就是主机内存。
        std::vector<double> ToHost(const Vector<double>& v)
        {
            const auto mirror = Kokkos::create_mirror_view_and_copy(HostMemory(), v.View());
            std::vector<double> values(mirror.extent(0));
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = mirror(i);
            }
            return values;
        }

        TEST(VectorTest, AllocatesRequestedSize)
        {
            const Vector<double> v(8);
            EXPECT_EQ(v.Size(), 8u);
            EXPECT_NE(v.Data(), nullptr);
        }

        TEST(VectorTest, DotAndNormAgreeWithScalarMath)
        {
            Vector<double> x(4);
            Kokkos::deep_copy(x.View(), 2.0);

            EXPECT_DOUBLE_EQ(x.Dot(x), 16.0);
            EXPECT_DOUBLE_EQ(x.SquaredNorm(), 16.0);
            EXPECT_DOUBLE_EQ(x.Norm(), 4.0);
        }

        TEST(VectorTest, AxpyAndScale)
        {
            Vector<double> x(3);
            Vector<double> y(3);
            Kokkos::deep_copy(x.View(), 1.0);
            Kokkos::deep_copy(y.View(), 3.0);

            y.Axpy(2.0, x);
            y.Scale(0.5);

            EXPECT_EQ(ToHost(y), std::vector<double>({2.5, 2.5, 2.5}));
        }

        TEST(VectorTest, CopyFromLeavesSourceUntouched)
        {
            Vector<double> source(2);
            Vector<double> target(2);
            Kokkos::deep_copy(source.View(), 4.0);
            Kokkos::deep_copy(target.View(), 0.0);

            target.CopyFrom(source);
            source.Scale(0.0);

            EXPECT_EQ(ToHost(target), std::vector<double>({4.0, 4.0}));
        }

        TEST(VectorTest, WrapsExistingViewWithoutCopy)
        {
            Kokkos::View<double*> storage("test::storage", 3);
            Kokkos::deep_copy(storage, 7.0);

            const Vector<double> wrapped(storage);
            storage(0) = 1.0;

            // 包装不复制数据：对底层 View 的修改对 Vector 可见。
            EXPECT_DOUBLE_EQ(wrapped.Dot(wrapped), 99.0);
        }

    } // namespace
} // namespace numerix
