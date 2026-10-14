# 集中依赖声明。Kokkos Core 与 Kokkos Kernels 都是必需依赖：
# Kokkos 提供执行空间、内存空间与 View，Kokkos Kernels 提供 portable BLAS 与稀疏容器。
# numerix 只在其上定义科学计算语义，不重写成熟 kernel（见 docs/DESIGN.md）。
if(PROJECT_IS_TOP_LEVEL)
    set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/.cache/cpm" CACHE PATH "CPM source cache")
endif()

set(CPM_DONT_UPDATE_MODULE_PATH ON)
set(CPM_DONT_CREATE_PACKAGE_LOCK ON)
if(NOT COMMAND CPMAddPackage)
    include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")
endif()

# Kokkos 与 Kokkos Kernels 必须版本同步，因此共用一个 tag。
set(NUMERIX_KOKKOS_VERSION "5.2.2" CACHE STRING "pinned Kokkos / Kokkos Kernels release tag")

# KokkosKernels 5.2.2 用 CMAKE_CXX_SIMULATE_ID == MSVC 判定"MSVC ABI"，但 Windows 上 clang++（GNU 驱动）
# 的 SIMULATE_ID 同样是 MSVC，于是它注入的是 MSVC 语法开关 /EHsc，而 GNU 驱动把 /EHsc 当成输入文件，
# 报 "no such file or directory: '/EHsc'"。KokkosKernels 是必需依赖，因此在配置阶段就给出可执行的替代方案，
# 而不是让错误在依赖内部以编译命令的形式出现。
if(NOT NUMERIX_USE_SYSTEM_KOKKOS
   AND WIN32
   AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    message(FATAL_ERROR
        "numerix requires Kokkos Kernels, which cannot be built with the clang++ (GNU) driver on Windows: "
        "KokkosKernels keys its MSVC flags on CMAKE_CXX_SIMULATE_ID (MSVC for both clang-cl and clang++ on "
        "Windows) and then injects the MSVC-driver flag /EHsc, which clang++ treats as an input file. "
        "Configure with the MSVC driver instead, e.g. -DCMAKE_CXX_COMPILER=clang-cl, or use an installed "
        "Kokkos Kernels with -DNUMERIX_USE_SYSTEM_KOKKOS=ON.")
endif()

if(NUMERIX_USE_SYSTEM_KOKKOS)
    find_package(Kokkos REQUIRED)
    find_package(KokkosKernels REQUIRED)
else()
    # Keep MSVC C4702 enabled in numerix: Kokkos 5.2.2 has non-exclusive
    # constexpr branches in has_all_zero_bits, which MSVC diagnoses even
    # through SYSTEM headers. The patch only makes those branches exclusive.
    set(_NUMERIX_KOKKOS_PATCH_ARGS)
    if(NUMERIX_KOKKOS_VERSION STREQUAL "5.2.2")
        set(_NUMERIX_KOKKOS_PATCH_ARGS PATCHES
            "${CMAKE_CURRENT_LIST_DIR}/patches/kokkos-5.2.2-unreachable.patch")
    endif()
    CPMAddPackage(
        NAME Kokkos
        SYSTEM YES
        GITHUB_REPOSITORY kokkos/kokkos
        GIT_TAG ${NUMERIX_KOKKOS_VERSION}
        ${_NUMERIX_KOKKOS_PATCH_ARGS}
        OPTIONS
        "Kokkos_ENABLE_SERIAL ON"
        "Kokkos_ENABLE_TESTS OFF"
        "Kokkos_ENABLE_EXAMPLES OFF"
        "Kokkos_ENABLE_BENCHMARKS OFF"
    )
    unset(_NUMERIX_KOKKOS_PATCH_ARGS)
    CPMAddPackage(
        NAME KokkosKernels
        SYSTEM YES
        GITHUB_REPOSITORY kokkos/kokkos-kernels
        GIT_TAG ${NUMERIX_KOKKOS_VERSION}
        OPTIONS
        "KokkosKernels_ENABLE_TESTS OFF"
        "KokkosKernels_ENABLE_EXAMPLES OFF"
        "KokkosKernels_ENABLE_BENCHMARKS OFF"
    )
    # KokkosKernels enables /bigobj for CUDA, but its CPU ETI sources also
    # exceed MSVC's COFF section limit in Debug builds (C1128).
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        target_compile_options(kokkoskernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        # Its batched SIMD header contains unguarded GCC diagnostic pragmas.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(kokkoskernels PRIVATE /wd4068)
        endif()
    endif()

endif()

if(NUMERIX_BUILD_TESTS AND NOT TARGET GTest::gtest)
    CPMFindPackage(
        NAME GTest
        SYSTEM YES
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.17.0
        OPTIONS
        "BUILD_GMOCK OFF"
        "INSTALL_GTEST OFF"
        "BUILD_SHARED_LIBS OFF"
        "gtest_force_shared_crt ON"
    )
    # googletest 自带 /WX，其 char8_t 打印路径在 clang 22 上触发 -Wcharacter-conversion。
    if(TARGET gtest AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
        target_compile_options(gtest PRIVATE $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-character-conversion>)
    endif()
endif()
