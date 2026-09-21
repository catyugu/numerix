#include <gtest/gtest.h>

#include <complex>
#include <sstream>
#include <string>
#include <type_traits>

#include <config.h>
#include <numerix/numerix.hpp>

namespace numerix {
    namespace {

        // 概念约束必须在编译期生效：不满足契约的类型不应被接受。
        static_assert(Scalar<double>);
        static_assert(Scalar<float>);
        static_assert(Scalar<std::complex<double>>);
        static_assert(!Scalar<int>);
        static_assert(!Scalar<bool>);
        static_assert(Complex<std::complex<float>>);
        static_assert(!Complex<double>);

        static_assert(std::is_same_v<RealOfT<double>, double>);
        static_assert(std::is_same_v<RealOfT<std::complex<float>>, float>);

        static_assert(VectorLike<StaticVector<double, 3>>);
        static_assert(VectorLike<Vector<double>>);
        static_assert(!VectorLike<double>);

        static_assert(MemorySpace<Memory>);
        static_assert(MemorySpace<HostMemory>);

        constexpr std::complex<double> kZ {3.0, 4.0};
        static_assert(SquaredMagnitude(kZ) == 25.0);
        static_assert(Conj(kZ) == std::complex<double>(3.0, -4.0));
        static_assert(Conj(2.0) == 2.0);

        constexpr StaticMatrix<double, 2, 3> MakeRectangular()
        {
            StaticMatrix<double, 2, 3> matrix;
            double value = 1.0;
            for (std::size_t i = 0; i < 2; ++i) {
                for (std::size_t j = 0; j < 3; ++j) {
                    matrix(i, j) = value;
                    value += 1.0;
                }
            }
            return matrix;
        }

        // 定长代数必须能整段在编译期求值。
        constexpr auto kRect = MakeRectangular();
        static_assert(kRect(0, 0) == 1.0);
        static_assert(kRect(1, 2) == 6.0);
        static_assert(kRect.Transpose()(2, 1) == 6.0);
        static_assert(StaticMatrix<double, 3, 3>::Identity().Trace() == 3.0);

        constexpr StaticVector<double, 3> kOnes {{1.0, 1.0, 1.0}};
        constexpr auto kProduct = kRect.Apply(kOnes);
        static_assert(kProduct[0] == 6.0);
        static_assert(kProduct[1] == 15.0);

        constexpr StaticVector<double, 3> kA {{3.0, 4.0, 0.0}};
        static_assert(kA.SquaredNorm() == 25.0);
        static_assert(kA.Dot(kA) == 25.0);
        static_assert((kA + kA)[0] == 6.0);
        static_assert((kA - kA)[0] == 0.0);
        static_assert((2.0 * kA)[1] == 8.0);

        TEST(StatusTest, RoundTripsThroughName)
        {
            EXPECT_TRUE(IsOk(Status::kOk));
            EXPECT_FALSE(IsOk(Status::kBreakdown));
            EXPECT_EQ(ToString(Status::kNotConverged), "not converged");
        }

        TEST(LoggerTest, FiltersBelowConfiguredLevel)
        {
            std::ostringstream sink;
            Logger logger(LogLevel::kWarn, sink);

            NUMERIX_LOG_INFO(logger, "hidden");
            NUMERIX_LOG_ERROR(logger, "visible");

            const std::string expected = std::string("[error] visible") + char(10);
            EXPECT_EQ(sink.str(), expected);
            EXPECT_EQ(logger.Level(), LogLevel::kWarn);
        }

        TEST(LoggerTest, MacrosCoverEveryLevel)
        {
            std::ostringstream sink;
            Logger logger(LogLevel::kDebug, sink);

            NUMERIX_LOG_DEBUG(logger, "d");
            NUMERIX_LOG_INFO(logger, "i");
            NUMERIX_LOG_WARN(logger, "w");
            NUMERIX_LOG_ERROR(logger, "e");

            const std::string newline(1, char(10));
#ifdef NUMERIX_DEBUG
            const std::string debug_line = "[debug] d" + newline;
#else
            const std::string debug_line; // DEBUG logging is compiled out in release builds
#endif
            const std::string expected = debug_line + "[info] i" + newline + "[warn] w" + newline + "[error] e" + newline;
            EXPECT_EQ(sink.str(), expected);
        }

        TEST(LoggerTest, DebugMacroIsElidedInReleaseBuilds)
        {
            std::ostringstream sink;
            Logger logger(LogLevel::kDebug, sink);

            NUMERIX_LOG_DEBUG(logger, "compiled out in release");

#ifdef NUMERIX_DEBUG
            EXPECT_EQ(sink.str(), std::string("[debug] compiled out in release") + char(10));
#else
            EXPECT_TRUE(sink.str().empty());
#endif
        }

        TEST(LoggerTest, DefaultLevelIsInfo)
        {
            EXPECT_EQ(Logger().Level(), LogLevel::kInfo);
            EXPECT_EQ(ToString(LogLevel::kDebug), "debug");
        }

        TEST(ConfigTest, GeneratedHeaderMatchesProjectVersion)
        {
            EXPECT_EQ(config::kProjectName, "Numerix");
            EXPECT_EQ(config::kVersion, "0.1.0");
            EXPECT_EQ(config::kVersionMajor, 0);
            EXPECT_EQ(config::kVersionMinor, 1);
            EXPECT_EQ(config::kVersionPatch, 0);
        }

        TEST(StaticAlgebraTest, NormIsNotConstexprButCorrect)
        {
            EXPECT_DOUBLE_EQ(kA.Norm(), 5.0);
        }

    } // namespace
} // namespace numerix
