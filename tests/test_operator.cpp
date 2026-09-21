#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include <numerix/numerix.hpp>

namespace numerix {
    namespace {

        constexpr std::size_t kSize = 64;

        Vector<double> MakeVector(double base, double step)
        {
            Vector<double> v(kSize);
            auto values = v.View();
            Kokkos::parallel_for(
                "test::Fill", Kokkos::RangePolicy<>(0, static_cast<int>(kSize)),
                KOKKOS_LAMBDA(int i) { values(i) = base + step * i; });
            return v;
        }

        std::vector<double> ToHost(const Vector<double>& v)
        {
            const auto mirror = Kokkos::create_mirror_view_and_copy(HostMemory(), v.View());
            std::vector<double> values(mirror.extent(0));
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = mirror(i);
            }
            return values;
        }

        Vector<double> LinearCombination(double alpha, const Vector<double>& x, double beta, const Vector<double>& y)
        {
            Vector<double> result(kSize);
            result.CopyFrom(x);
            result.Scale(alpha);
            result.Axpy(beta, y);
            return result;
        }

        // 测试算子：y = alpha * x
        class ScaleOperator {
        public:
            explicit ScaleOperator(double alpha) : alpha_(alpha) { }

            void Apply(const Vector<double>& x, Vector<double>& y) const
            {
                y.CopyFrom(x);
                y.Scale(alpha_);
            }

        private:
            double alpha_;
        };

        // 测试算子：y = diag(d) * x
        class DiagonalOperator {
        public:
            explicit DiagonalOperator(Vector<double> diagonal) : diagonal_(std::move(diagonal)) { }

            void Apply(const Vector<double>& x, Vector<double>& y) const
            {
                const auto diagonal = diagonal_.View();
                const auto input = x.View();
                auto output = y.View();
                Kokkos::parallel_for(
                    "test::DiagonalOperator", Kokkos::RangePolicy<>(0, static_cast<int>(x.Size())),
                    KOKKOS_LAMBDA(int i) { output(i) = diagonal(i) * input(i); });
            }

        private:
            Vector<double> diagonal_;
        };

        static_assert(LinearOperator<ScaleOperator, Vector<double>>);
        static_assert(LinearOperator<DiagonalOperator, Vector<double>>);
        static_assert(Preconditioner<DiagonalOperator, Vector<double>>);

        TEST(OperatorPropertyTest, LinearityHoldsForDiagonalOperator)
        {
            const DiagonalOperator op(MakeVector(1.0, 0.5));
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);
            const double alpha = 1.5;
            const double beta = -0.5;

            const Vector<double> combined = LinearCombination(alpha, x, beta, y);
            Vector<double> lhs(kSize);
            op.Apply(combined, lhs);

            Vector<double> ax(kSize);
            Vector<double> by(kSize);
            Vector<double> rhs(kSize);
            op.Apply(x, ax);
            op.Apply(y, by);
            rhs.CopyFrom(ax);
            rhs.Scale(alpha);
            rhs.Axpy(beta, by);

            const auto lhs_host = ToHost(lhs);
            const auto rhs_host = ToHost(rhs);
            for (std::size_t i = 0; i < kSize; ++i) {
                EXPECT_NEAR(lhs_host[i], rhs_host[i], 1e-12);
            }
        }

        TEST(OperatorPropertyTest, SymmetryHoldsForDiagonalOperator)
        {
            const DiagonalOperator op(MakeVector(0.5, 0.25));
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);

            Vector<double> ax(kSize);
            Vector<double> ay(kSize);
            op.Apply(x, ax);
            op.Apply(y, ay);

            EXPECT_NEAR(x.Dot(ay), y.Dot(ax), 1e-12);
        }

        TEST(OperatorPropertyTest, PositiveDefinitenessHoldsForPositiveDiagonal)
        {
            const DiagonalOperator op(MakeVector(0.5, 0.25));
            const Vector<double> x = MakeVector(2.0, -0.25);

            Vector<double> ax(kSize);
            op.Apply(x, ax);

            EXPECT_GT(x.Dot(ax), 0.0);
        }

        TEST(AnyLinearOperatorTest, MatchesDirectApplication)
        {
            const DiagonalOperator op(MakeVector(1.0, 0.5));
            const Vector<double> x = MakeVector(2.0, -0.25);

            const AnyLinearOperator<Vector<double>> erased(op);

            Vector<double> direct(kSize);
            Vector<double> through_erasure(kSize);
            op.Apply(x, direct);
            erased.Apply(x, through_erasure);

            EXPECT_EQ(ToHost(direct), ToHost(through_erasure));
        }

        TEST(PipelineTest, MatchesSequentialApplication)
        {
            const DiagonalOperator diagonal(MakeVector(1.0, 0.5));
            const ScaleOperator scale(2.0);
            const Vector<double> x = MakeVector(2.0, -0.25);

            const auto pipeline = Compose<Vector<double>>(kSize, diagonal, scale);

            Vector<double> through_pipeline(kSize);
            pipeline.Apply(x, through_pipeline);

            Vector<double> step_one(kSize);
            Vector<double> step_two(kSize);
            diagonal.Apply(x, step_one);
            scale.Apply(step_one, step_two);

            EXPECT_EQ(ToHost(through_pipeline), ToHost(step_two));
        }

        TEST(PipelineTest, ComposedOperatorIsStillLinear)
        {
            const auto pipeline = Compose<Vector<double>>(kSize, DiagonalOperator(MakeVector(1.0, 0.5)), ScaleOperator(2.0));
            const Vector<double> x = MakeVector(2.0, -0.25);
            const Vector<double> y = MakeVector(-1.0, 0.75);
            const double alpha = 1.5;
            const double beta = -0.5;

            const Vector<double> combined = LinearCombination(alpha, x, beta, y);
            Vector<double> lhs(kSize);
            pipeline.Apply(combined, lhs);

            Vector<double> ax(kSize);
            Vector<double> ay(kSize);
            Vector<double> rhs(kSize);
            pipeline.Apply(x, ax);
            pipeline.Apply(y, ay);
            rhs.CopyFrom(ax);
            rhs.Scale(alpha);
            rhs.Axpy(beta, ay);

            const auto lhs_host = ToHost(lhs);
            const auto rhs_host = ToHost(rhs);
            for (std::size_t i = 0; i < kSize; ++i) {
                EXPECT_NEAR(lhs_host[i], rhs_host[i], 1e-12);
            }
        }

    } // namespace
} // namespace numerix
