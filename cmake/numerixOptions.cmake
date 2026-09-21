# 使用需求（INTERFACE）：所有使用 numerix 的目标都必须满足的编译条件。
add_library(numerix_options INTERFACE)

target_compile_features(numerix_options INTERFACE cxx_std_20)

set(_NUMERIX_GNU_FRONTEND $<CXX_COMPILER_FRONTEND_VARIANT:GNU>)
set(_NUMERIX_MSVC_FRONTEND $<CXX_COMPILER_FRONTEND_VARIANT:MSVC>)

target_compile_options(numerix_options INTERFACE
    $<${_NUMERIX_MSVC_FRONTEND}:
    /permissive-
    /utf-8
    /bigobj
    /Zc:__cplusplus
    >
)

target_compile_options(numerix_options INTERFACE
    $<$<AND:$<PLATFORM_ID:linux>,${_NUMERIX_GNU_FRONTEND}>:
    -fPIC
    >
)

target_link_options(numerix_options INTERFACE
    $<$<AND:$<PLATFORM_ID:linux>,${_NUMERIX_GNU_FRONTEND}>:
    -rdynamic # Required by stack traceback
    >
)

target_compile_definitions(numerix_options INTERFACE
    $<$<CONFIG:Debug>:NUMERIX_DEBUG>
)

target_compile_options(numerix_options INTERFACE
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Debug>>:-O0>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Debug>>:-g3>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Release>>:-O2>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Release>>:-fno-omit-frame-pointer>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:RelWithDebInfo>>:-O2>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:RelWithDebInfo>>:-g>
    $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>
)

option(NUMERIX_WARNINGS_AS_ERRORS "treat compiler warnings in numerix targets as errors" ON)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)

# 警告属于 PRIVATE 需求：只作用于 numerix 自己的目标，不通过 PUBLIC 传播给使用方。
add_library(numerix_warnings INTERFACE)

if(NUMERIX_WARNINGS_AS_ERRORS)
    set(_NUMERIX_WERROR_GNU -Werror)
    set(_NUMERIX_WERROR_MSVC /WX)
else()
    set(_NUMERIX_WERROR_GNU "")
    set(_NUMERIX_WERROR_MSVC "")
endif()

target_compile_options(numerix_warnings INTERFACE
    $<${_NUMERIX_GNU_FRONTEND}:
    -Wall
    -Wextra
    -Wpedantic
    -fno-strict-aliasing
    ${_NUMERIX_WERROR_GNU}
    >
    $<${_NUMERIX_MSVC_FRONTEND}:
    /W4
    ${_NUMERIX_WERROR_MSVC}
    >
)

if(ENABLE_COVERAGE)
    target_compile_options(numerix_warnings INTERFACE
        $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Debug>>:--coverage>
    )
    target_link_options(numerix_warnings INTERFACE
        $<$<AND:${_NUMERIX_GNU_FRONTEND},$<CONFIG:Debug>>:--coverage>
    )
endif()

unset(_NUMERIX_GNU_FRONTEND)
unset(_NUMERIX_MSVC_FRONTEND)
unset(_NUMERIX_WERROR_GNU)
unset(_NUMERIX_WERROR_MSVC)
