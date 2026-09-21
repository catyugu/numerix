cmake_minimum_required(VERSION 3.24)

project(Numerix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# numerix 作为子项目时不得污染使用方：构建类型、并行度、编译数据库与优化级别都是
# top-level developer/build policy，只在 numerix 自己作为顶层项目时才设置。
if(PROJECT_IS_TOP_LEVEL)
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(CPM_SOURCE_CACHE "${PROJECT_SOURCE_DIR}/.cache/cpm" CACHE PATH "CPM cache to avoid repeated fetching")

# 子项目默认安静：测试、示例与基准是 numerix 自己的开发目标，不进入使用方的构建。
option(NUMERIX_BUILD_EXAMPLES "build examples" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_BUILD_TESTS "build tests" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_BUILD_BENCHMARKS "build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_USE_SYSTEM_KOKKOS "use installed Kokkos and Kokkos Kernels instead of fetching them" OFF)

include(GNUInstallDirs)

configure_file(${PROJECT_SOURCE_DIR}/cmake/config.h.in ${PROJECT_BINARY_DIR}/config.h)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

cmake_host_system_information(RESULT CPU_NUMS QUERY NUMBER_OF_LOGICAL_CORES)
message(STATUS "CPU NUMS on current device is:${CPU_NUMS}")
math(EXPR JOBS "${CPU_NUMS} + ${CPU_NUMS} / 2")
set(CMAKE_BUILD_PARALLEL_LEVEL ${JOBS} CACHE STRING "parallel compiling thread num")
message(STATUS "Compiler default work jobs: ${CMAKE_BUILD_PARALLEL_LEVEL}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(cmake/numerixOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)

if(NUMERIX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NUMERIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NUMERIX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/numerixConfig.cmake.in
    "${PROJECT_BINARY_DIR}/numerixConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/numerixConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
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
