# 集中依赖声明。Kokkos 是唯一必需第三方依赖；其余后端一律通过 adapter 接入（见 docs/DESIGN.md）。
include(CPM)

set(NUMERIX_KOKKOS_VERSION "5.2.2" CACHE STRING "pinned Kokkos release tag")

if(NUMERIX_USE_SYSTEM_KOKKOS)
    find_package(Kokkos REQUIRED)
else()
    CPMAddPackage(
        NAME Kokkos
        GITHUB_REPOSITORY kokkos/kokkos
        GIT_TAG ${NUMERIX_KOKKOS_VERSION}
        OPTIONS
        "Kokkos_ENABLE_SERIAL ON"
        "Kokkos_ENABLE_TESTS OFF"
        "Kokkos_ENABLE_EXAMPLES OFF"
        "Kokkos_ENABLE_BENCHMARKS OFF"
    )
endif()

if(NUMERIX_BUILD_TESTS)
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.17.0
        OPTIONS
        "BUILD_GMOCK OFF"
        "INSTALL_GTEST OFF"
    )
    # googletest 自带 /WX，其 char8_t 打印路径在 clang 22 上触发 -Wcharacter-conversion。
    if(TARGET gtest)
        target_compile_options(gtest PRIVATE $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-character-conversion>)
    endif()
endif()
