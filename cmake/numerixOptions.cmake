# Development policy is private to numerix targets and absent from the export.
add_library(numerix_options INTERFACE)
option(NUMERIX_WARNINGS_AS_ERRORS "Treat warnings in numerix targets as errors" ${PROJECT_IS_TOP_LEVEL})
option(NUMERIX_ENABLE_COVERAGE "Enable Debug coverage instrumentation" OFF)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(numerix_options INTERFACE
        /W4 /permissive- /utf-8 /bigobj /Zc:__cplusplus)
    if(NUMERIX_WARNINGS_AS_ERRORS)
        target_compile_options(numerix_options INTERFACE /WX)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(numerix_options INTERFACE -Wall -Wextra -Wpedantic)
    if(NUMERIX_WARNINGS_AS_ERRORS)
        target_compile_options(numerix_options INTERFACE -Werror)
    endif()
    if(PROJECT_IS_TOP_LEVEL)
        target_compile_options(numerix_options INTERFACE
            $<$<CONFIG:Debug>:-g3>
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>)
        target_link_options(numerix_options INTERFACE
            $<$<PLATFORM_ID:Linux>:-rdynamic>)
    endif()
    if(NUMERIX_ENABLE_COVERAGE)
        target_compile_options(numerix_options INTERFACE $<$<CONFIG:Debug>:--coverage>)
        target_link_options(numerix_options INTERFACE $<$<CONFIG:Debug>:--coverage>)
    endif()
elseif(NUMERIX_ENABLE_COVERAGE)
    message(FATAL_ERROR "Coverage requires a GCC or Clang GNU frontend")
endif()
if(NUMERIX_ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    message(FATAL_ERROR "Coverage requires a GCC or Clang GNU frontend")
endif()
