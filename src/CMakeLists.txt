add_library(numerix STATIC
    diagnostics/logger.cpp
)

add_library(numerix::numerix ALIAS numerix)

target_compile_features(numerix PUBLIC cxx_std_20)
target_compile_definitions(numerix PUBLIC $<$<CONFIG:Debug>:NUMERIX_DEBUG>)

target_include_directories(numerix PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# 使用方可能把 numerix 静态库链进自己的共享库，因此 numerix 自己带 PIC；
# 这是 numerix 目标的属性，不是强加给使用方的编译选项。
set_target_properties(numerix PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(numerix
    PUBLIC
        Kokkos::kokkos
        Kokkos::kokkoskernels
    PRIVATE
        $<BUILD_INTERFACE:numerix_options>
)

if(NUMERIX_INSTALL)
    install(TARGETS numerix EXPORT numerixTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/numerix
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES "${PROJECT_BINARY_DIR}/include/numerix/config.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/numerix)
    install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()
