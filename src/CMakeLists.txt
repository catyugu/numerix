add_library(numerix STATIC
    diagnostics/logger.cpp
)

target_include_directories(numerix PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(numerix PRIVATE ${PROJECT_BINARY_DIR})

# 使用方可能把 numerix 静态库链进自己的共享库，因此 numerix 自己带 PIC；
# 这是 numerix 目标的属性，不是强加给使用方的编译选项。
set_target_properties(numerix PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(numerix
    PUBLIC
        numerix_options
        Kokkos::kokkos
        Kokkos::kokkoskernels
    PRIVATE
        numerix_warnings
)

install(TARGETS numerix EXPORT numerixTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 使用方/构建选项接口目标必须与 numerix 同属一个导出集，否则 install(EXPORT) 会报缺失依赖。
install(TARGETS numerix_options numerix_warnings EXPORT numerixTargets)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/numerix
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
