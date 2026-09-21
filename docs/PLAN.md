# numerix 实施计划

原则：小步提交，每步可独立验证；先澄清依赖方向与数据契约，再动算法。
所有结论必须由真实运行结果支撑。

## 阶段 0：工程骨架（已完成）

交付：

- 顶层 CMake + `cmake/numerixOptions.cmake` + `cmake/Dependencies.cmake` + 安装导出
- 目录骨架 `include/`、`src/`、`tests/`、`examples/`、`benchmarks/`
- core / execution / memory / algebra / operator / diagnostics 的最小可用实现
- 测试（概念静态断言 + 线性性/对称性/正定性 + 流水线一致性）
- 示例与基准各一个

验收（本机实测：clang-cl 22 + Ninja，Kokkos 5.2.2 由 CPM 拉取，Serial 后端）：

- [x] `cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Debug` 与 `-DCMAKE_BUILD_TYPE=Release` 均配置成功
- [x] `cmake --build <build>` 零警告（`/WX` 与 `-Werror` 均生效）
- [x] `ctest --test-dir <build>` 18/18 通过（Debug 与 Release 双配置）
- [x] 示例输出与解析解一致（`dot(y, y) = 144`）
- [x] 基准同时给出 numerix kernel、裸 Kokkos kernel 与手写循环的带宽，并带可校验的 checksum
- [x] `cmake --install` 后，独立使用方项目通过 `find_package(numerix)` 编译并运行成功

实测结论：Serial 后端下 numerix 的 axpy 带宽与裸 Kokkos kernel 相当，手写循环略快
（Kokkos Serial 的逐元素调用比手写循环慢约 10%–20%，属 Kokkos 序列后端特性，不是 numerix 的额外开销）。
复现：`./build-release/benchmarks/numerix_axpy_bench.exe`。

## 阶段 1：动态代数补齐

- `MultiVector`：块 Krylov、多右端、POD/ROM 的基础容器
- 稀疏算子包装 `SparseOperator`（基于 `KokkosSparse::CrsMatrix`，不自行发明 CSR）
- 向量空间/内积语义的统一约定（实数与复数）

验收：`MultiVector` 的列内积与逐列 `Vector::Dot` 一致；稀疏算子与逐行手工计算结果一致。

## 阶段 2：线性求解器

- 接口：`SolveResult Solve(const A&, const X& b, X& x)`
- 迭代法：CG、MINRES、GMRES、FGMRES、BiCGStab
- 预条件子：Identity、Jacobi、BlockJacobi、ILU(0)
- 收敛准则：相对残差、绝对残差与最大迭代次数，`ConvergenceMonitor` 记录残差历史

验收：对已知谱的对称正定系统，CG 的迭代次数与理论估计同阶；GMRES 在 m >= n 时给出直接解；
预条件子开启后迭代次数下降。

不实现：AMG、稀疏直接法（交由 PETSc / Hypre / Ginkgo / MUMPS）。

## 阶段 3：非线性与时间积分

- `NonlinearOperator` 与 `ApplyJ(x, v)`（优先 matrix-free Jacobian-向量积）
- Newton、Newton-Krylov、Picard、Anderson
- 时间积分：Explicit RK、SSPRK、BDF、SDIRK、IMEX，与 PDE 解耦

验收：Jacobian 的有限差分一致性测试；Newton 对已知非线性问题二次收敛；
时间积分在制造解上的收敛阶与理论阶一致。

## 阶段 4：分布式与后端

- `Communicator`、`DistributedVector`、`Partition`、`HaloExchange`（MPI 可选）
- 后端 adapter：eigen / blas / lapack / hdf5 / mpi / petsc / ginkgo
- 可选依赖不得进入核心头文件

验收：单进程与多进程结果一致；adapter 关闭时核心库仍可独立构建。

## 阶段 5：诊断与性能

- `Timer`、`Profiler`、`ResidualHistory`、`ConvergenceMonitor`
- 基准扩展到稀疏矩阵-向量乘、点积、流水线的内存带宽与强/弱扩展性

验收：基准给出可复现的带宽与扩展性数据；诊断开销在热路径中可关闭。

## 待办与风险

- 本机为 clang-cl + Ninja，需在 GCC/OpenMP 环境复验构建与测试。
- Kokkos 默认开启 Serial；OpenMP/CUDA 后端需在对应环境下复验（`Kokkos_ENABLE_OPENMP` 等）。
- 复数标量的求解器测试待阶段 2 补充。
- 格式检查依赖 `uv tool run --from clang-format clang-format`，本机未安装独立 clang-format 二进制。
- 仓库尚未初始化 git；首次提交前需确认远程地址与许可证归属。
