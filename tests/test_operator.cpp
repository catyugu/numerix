#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include <numerix/numerix.hpp>

#include "test_support.hpp"

namespace numerix {
    namespace {

        using test_support::Exec;
        using test_support::FillRamp;
        using test_support::index_type;
        using test_support::RangeOf;
        using test_support::ToHost;

        constexpr std::size_t kSize = 64;
        using View = Kokkos::View<double*, DefaultMemorySpace>;

        Vector<double> MakeVector(double base, double step)
        {
            Vector<double> v(kSize);
            FillRamp(v, base, step);
            return v;
        }

        // 测试算子：y = alpha * A x + beta * y，其中 A = diag(d)。
        // 融合形式一次 kernel 完成，省掉一次额外的内存往返（KokkosSparse::spmv 也是这种形式）。
        // 输出 View 按 Kokkos 约定用 const 引用传：句柄是 const 的，数据仍可写。
        class DiagonalOperator {
        public:
            explicit DiagonalOperator(View diagonal) : diagonal_(std::move(diagonal)) { }

            template <class Exec, class X, class Y>
            void Apply(const Exec& exec, const X& x, const Y& y) const
            {
                Apply(exec, 1.0, x, 0.0, y);
            }

            template <class Exec, class X, class Y>
            void Apply(const Exec& exec, double alpha, const X& x, double beta, const Y& y) const
            {
                const auto diagonal = diagonal_;
                Kokkos::parallel_for(
                    "test::DiagonalOperator::Apply", RangeOf(exec, x),
                    KOKKOS_LAMBDA(index_type i) { y(i) = alpha * diagonal(i) * x(i) + beta * y(i); });
            }

        private:
            View diagonal_;
        };

        // 只有 y = A x 的算子：用来确认 ScaledLinearOperatorFor 不会误判。
        class PlainScaleOperator {
        public:
            explicit PlainScaleOperator(double alpha) : alpha_(alpha) { }

            template <class Exec, class X, class Y>
            void Apply(const Exec& exec, const X& x, const Y& y) const
            {
                Kokkos::deep_copy(exec, y, x);
                Scale(exec, alpha_, y);
            }

        private:
            double alpha_;
        };

        struct NotAnOperator { };

        static_assert(LinearOperatorFor<DiagonalOperator, DefaultExecutionSpace, View>);
        static_assert(ScaledLinearOperatorFor<DiagonalOperator, DefaultExecutionSpace, View>);
        static_assert(LinearOperatorFor<PlainScaleOperator, DefaultExecutionSpace, View>);
        static_assert(!ScaledLinearOperatorFor<PlainScaleOperator, DefaultExecutionSpace, View>);
        static_assert(!LinearOperatorFor<NotAnOperator, DefaultExecutionSpace, View>);

        TEST(OperatorPropertyTest, LinearityHoldsForDiagonalOperator)
        {
            Vector<double> diagonal = MakeVector(1.0, 0.5);
            const DiagonalOperator op(diagonal.View());
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);
            constexpr double kAlpha = 1.5;
            constexpr double kBeta = -0.5;

            // A(alpha x + beta y)
            Vector<double> combined = Clone(Exec(), x);
            Scale(Exec(), kAlpha, combined.View());
            Axpy(Exec(), kBeta, y.View(), combined.View());
            Vector<double> lhs(kSize);
            op.Apply(Exec(), combined.View(), lhs.View());

            // alpha (A x) + beta (A y)
            Vector<double> rhs(kSize);
            Vector<double> ax(kSize);
            Vector<double> ay(kSize);
            op.Apply(Exec(), x.View(), ax.View());
            op.Apply(Exec(), y.View(), ay.View());
            Axpy(Exec(), kAlpha, ax.View(), rhs.View());
            Axpy(Exec(), kBeta, ay.View(), rhs.View());

            const auto lhs_host = ToHost(lhs);
            const auto rhs_host = ToHost(rhs);
            for (std::size_t i = 0; i < kSize; ++i) {
                EXPECT_NEAR(lhs_host[i], rhs_host[i], 1e-12);
            }
        }

        TEST(OperatorPropertyTest, SymmetryHoldsForDiagonalOperator)
        {
            Vector<double> diagonal = MakeVector(0.5, 0.25);
            const DiagonalOperator op(diagonal.View());
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);

            Vector<double> ax(kSize);
            Vector<double> ay(kSize);
            op.Apply(Exec(), x.View(), ax.View());
            op.Apply(Exec(), y.View(), ay.View());

            EXPECT_NEAR(Dot(Exec(), x.View(), ay.View()), Dot(Exec(), y.View(), ax.View()), 1e-12);
        }

        TEST(OperatorPropertyTest, PositiveDefinitenessHoldsForPositiveDiagonal)
        {
            Vector<double> diagonal = MakeVector(0.5, 0.25);
            const DiagonalOperator op(diagonal.View());
            const Vector<double> x = MakeVector(2.0, -0.25);

            Vector<double> ax(kSize);
            op.Apply(Exec(), x.View(), ax.View());

            EXPECT_GT(Dot(Exec(), x.View(), ax.View()), 0.0);
        }

        // 融合形式必须与非融合组合在数学上一致：y = alpha * A x + beta * y。
        TEST(ScaledOperatorTest, FusedApplyMatchesUnfusedCombination)
        {
            Vector<double> diagonal = MakeVector(1.0, 0.5);
            const DiagonalOperator op(diagonal.View());
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);
            constexpr double kAlpha = 1.5;
            constexpr double kBeta = -0.5;

            Vector<double> fused = Clone(Exec(), y);
            op.Apply(Exec(), kAlpha, x.View(), kBeta, fused.View());

            Vector<double> reference = Clone(Exec(), y);
            Scale(Exec(), kBeta, reference.View());
            Vector<double> ax(kSize);
            op.Apply(Exec(), x.View(), ax.View());
            Axpy(Exec(), kAlpha, ax.View(), reference.View());

            const auto fused_host = ToHost(fused);
            const auto reference_host = ToHost(reference);
            for (std::size_t i = 0; i < kSize; ++i) {
                EXPECT_NEAR(fused_host[i], reference_host[i], 1e-12);
            }
        }

        // 不带融合形式的算子仍然满足基础契约：求解器可以按自己的 workspace 做回退。
        TEST(OperatorTest, PlainOperatorSatisfiesBaseContract)
        {
            const PlainScaleOperator op(2.0);
            const Vector<double> x = MakeVector(2.0, -0.25);

            Vector<double> ax(kSize);
            op.Apply(Exec(), x.View(), ax.View());

            EXPECT_DOUBLE_EQ(Dot(Exec(), x.View(), ax.View()), 2.0 * Dot(Exec(), x.View(), x.View()));
        }

    } // namespace
} // namespace numerix
