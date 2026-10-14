cmake_minimum_required(VERSION 3.26)

if(TARGET numerix::numerix)
    return()
endif()

project(Numerix LANGUAGES CXX)

set(NUMERIX_VERSION_MAJOR 0)
set(NUMERIX_VERSION_MINOR 1)
set(NUMERIX_VERSION_PATCH 0)
set(NUMERIX_VERSION "${NUMERIX_VERSION_MAJOR}.${NUMERIX_VERSION_MINOR}.${NUMERIX_VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# numerix 作为子项目时不得污染使用方：构建类型、并行度、编译数据库与优化级别都是
# top-level developer/build policy，只在 numerix 自己作为顶层项目时才设置。
if(PROJECT_IS_TOP_LEVEL)
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    endif()
    if(NOT CMAKE_CONFIGURATION_TYPES)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# 子项目默认安静：测试、示例与基准是 numerix 自己的开发目标，不进入使用方的构建。
option(NUMERIX_BUILD_EXAMPLES "build examples" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_BUILD_TESTS "build tests" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_BUILD_BENCHMARKS "build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_USE_SYSTEM_KOKKOS "use installed Kokkos and Kokkos Kernels instead of fetching them" OFF)

option(NUMERIX_INSTALL "Generate install rules for numerix" ON)

include(GNUInstallDirs)

configure_file(cmake/config.h.in "${PROJECT_BINARY_DIR}/include/numerix/config.h" @ONLY)

include(cmake/numerixOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

if(NUMERIX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NUMERIX_BUILD_TESTS)
    if(PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_subdirectory(tests)
endif()

if(NUMERIX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(NUMERIX_INSTALL)
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        cmake/numerixConfig.cmake.in
        "${PROJECT_BINARY_DIR}/numerixConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
    )
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/numerixConfigVersion.cmake"
        VERSION ${NUMERIX_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        "${PROJECT_BINARY_DIR}/numerixConfig.cmake"
        "${PROJECT_BINARY_DIR}/numerixConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
    )
    install(EXPORT numerixTargets
        NAMESPACE numerix::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
    )
endif()
