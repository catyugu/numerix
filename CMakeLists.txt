cmake_minimum_required(VERSION 3.24)

project(Numerix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache/cpm" CACHE PATH "CPM cache to avoid repeated fetching")

option(NUMERIX_BUILD_EXAMPLES "build examples" ON)
option(NUMERIX_BUILD_TESTS "build tests" ON)
option(NUMERIX_BUILD_BENCHMARKS "build benchmarks" ON)
option(NUMERIX_USE_SYSTEM_KOKKOS "use an installed Kokkos package instead of fetching one" OFF)

include(GNUInstallDirs)

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_BINARY_DIR}/config.h)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
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
    "${CMAKE_CURRENT_BINARY_DIR}/numerixConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/numerixConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/numerixConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/numerixConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
)
install(EXPORT numerixTargets
    NAMESPACE numerix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerix
)
