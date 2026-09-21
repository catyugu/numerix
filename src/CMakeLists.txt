add_library(numerix STATIC
    diagnostics/logger.cpp
)

target_include_directories(numerix PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(numerix PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(numerix
    PUBLIC
        numerix_options
        Kokkos::kokkos
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
