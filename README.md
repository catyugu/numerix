# numerix

C++20 + Kokkos-native 的通用科学计算基础库。

numerix 不提供 FEM/FVM/BEM 框架，而提供所有离散方法共享的数值执行模型：执行空间、内存空间、
稠密/定长代数、线性算子、求解器与诊断。Kokkos 是 numerix 的机器模型，numerix 只在其上提供
科学计算语义；BLAS 与稀疏容器交给 Kokkos Kernels，不自行重写。

## 要求

- CMake >= 3.26
- 支持 C++20 的编译器（GCC / Clang / MSVC）
- 推荐生成器：Ninja

Windows 上必须使用 MSVC 驱动（clang-cl 或 cl.exe）：KokkosKernels 5.2.2 对 clang++（GNU 驱动）
会注入 MSVC 语法开关 `/EHsc`，clang++ 把它当成输入文件。配置阶段会直接给出该提示，不会等到编译。
PATH 里同时有 clang++ 与 clang-cl 时（如 LLVM 发行包），显式指定编译器，并且 C 与 C++ 都指定
（googletest 的 `project()` 未声明语言，会顺带探测 C 编译器）：

```bash
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
```

必需依赖是 Kokkos Core 与 Kokkos Kernels（同 tag，版本固定在 `cmake/Dependencies.cmake`），
默认由 CPM 拉取。

## 快速开始

```bash
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

示例与基准：

```bash
./build/examples/numerix_axpy
./build/benchmarks/numerix_axpy_bench
```

若本机已安装 Kokkos 与 Kokkos Kernels，可改用系统包：

```bash
cmake -S . -B build -G "Ninja" -DNUMERIX_USE_SYSTEM_KOKKOS=ON
```

## 安装与使用

```bash
cmake --install build --prefix <prefix>
```

使用方：

```cmake
find_package(numerix REQUIRED)
target_link_libraries(my_target PRIVATE numerix::numerix)
```

注意：

- 安装导出会把 numerix 与 CPM 拉取的 Kokkos / Kokkos Kernels 一起安装；若 numerix 是用
  `NUMERIX_USE_SYSTEM_KOKKOS=ON` 构建的，使用方需自行提供这两个包。
- MSVC/clang-cl 下 ABI 与 Debug/Release 相关，使用方的构建配置必须与 numerix 安装时一致。

## 开发命令

```bash
# 格式化（仓库根目录 .clang-format 为准）
uv tool run --from clang-format clang-format -i include/numerix/*.hpp include/numerix/*/*.hpp \
    src/*/*.cpp tests/*.cpp examples/*.cpp benchmarks/*.cpp
```

## 构建选项

| 选项                         | 默认 | 含义                                     |
| ---------------------------- | ---- | ---------------------------------------- |
| `NUMERIX_BUILD_EXAMPLES`     | ON   | 构建示例                                 |
| `NUMERIX_BUILD_TESTS`        | ON   | 构建测试（需要 googletest，由 CPM 拉取） |
| `NUMERIX_BUILD_BENCHMARKS`   | ON   | 构建基准程序                             |
| `NUMERIX_USE_SYSTEM_KOKKOS`  | OFF  | 使用已安装的 Kokkos 而非 CPM 拉取        |
| `NUMERIX_WARNINGS_AS_ERRORS` | ON   | numerix 自身目标把警告视为错误           |
| `NUMERIX_INSTALL`            | ON   | 生成 numerix 安装与包导出规则           |
| `NUMERIX_ENABLE_COVERAGE`    | OFF  | GCC/Clang GNU 前端的 Debug 覆盖率       |

作为子项目时，示例、测试、基准与警告报错默认关闭；构建类型、优化级别、并行度由父项目决定。
构建树和安装包都提供 `numerix::numerix`。当前 numerix 仍为静态库，不受 `BUILD_SHARED_LIBS` 影响。
`NUMERIX_INSTALL=OFF` 只关闭 numerix 的安装规则，依赖的安装规则由依赖自身控制。
生成的版本头文件使用 `<numerix/config.h>`，构建树与安装包的路径一致。
0.x 版本按 minor 版本判断兼容性。

## 持续集成

GitHub Actions 在 Linux（GCC、Clang）、macOS（AppleClang）、Windows（MSVC）上分别检查
Debug 与 Release，使用 CMake 3.26.4 验证最低版本。每个任务构建全部测试、示例与基准，运行 CTest，
再验证移动安装目录后的 `find_package` 消费者，以及使用系统 Kokkos 的 `add_subdirectory` 集成。
后者还检查父项目设置不被修改、重复引入可复用目标、开发目标默认关闭。

本地运行相同步骤：

```bash
python ci/run_checks.py --config Debug
python ci/run_checks.py --config Release --build-root build-ci-release
```

可用 `--generator "Ninja Multi-Config"` 指定与 CI 一致的多配置生成器；默认使用平台原生生成器。
Windows 使用 Ninja 时需在 Developer PowerShell / Command Prompt 内运行，CI 会自动初始化 MSVC 环境。

## 目录结构

```bash
numerix/
├── cmake/            构建配置与集中依赖声明
├── docs/             设计文档与实施计划
├── include/numerix/  公共头文件
├── src/              库实现
├── tests/            测试
├── examples/         示例
└── benchmarks/       基准
```

## 文档

- `AGENTS.md`：项目概述与宪法（编码约定）
- `docs/DESIGN.md`：设计文档（目标、模块边界、约定、决策记录）
- `docs/PLAN.md`：分阶段实施计划与验收标准

## 许可证

MIT，见 `LICENSE`。
