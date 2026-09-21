# numerix 实施计划

原则：小步提交，每步可独立验证；先澄清依赖方向与数据契约，再动算法。
所有结论必须由真实运行结果支撑。

## 阶段 0：工程骨架（已完成）

交付：

- 顶层 CMake + `cmake/numerixOptions.cmake` + `cmake/Dependencies.cmake` + 安装导出
- 目录骨架 `include/`、`src/`、`tests/`、`examples/`、`benchmarks/`
- core / execution / memory / algebra / operator / diagnostics 的最小可用实现
- 测试（概念静态断言 + 线性性/对称性/正定性 + 融合算子语义 + 所有权与 const 语义）
- 示例与基准各一个

## 阶段 0.5：Kokkos-native 纠偏（已完成）

在动 `MultiVector` / 稀疏算子 / 求解器之前先收敛执行模型（理由见 `docs/DESIGN.md` D-1 ~ D-10）：

- 标量体系改为 `ScalarTraits<T>` 定义，复数用 `Kokkos::complex`，全部 `KOKKOS_INLINE_FUNCTION`
- `Vector` 重做为 owning、move-only 存储，`const` 只交出只读 View，复制必须显式 `Clone` / `DeepCopy`
- 代数运算改为 `Scale/Axpy/Dot/Norm2(exec, ...)`，作用在 `Kokkos::View` 上，kernel 由 Kokkos Kernels 提供
- `LinearOperatorFor<Op, Exec, X, Y>` 携带 execution-space instance；融合形式用 `ScaledLinearOperatorFor`
- 删除 `Pipeline`、`AnyLinearOperator`、通用 `Status`、`VectorLike`、`Device`/`Memory` 别名与 int 索引窄化
- CMake 使用需求收缩；Kokkos Kernels 成为与 Kokkos 同 tag 的必需依赖

验收（本机实测：clang-cl + Ninja，Kokkos / Kokkos Kernels 5.2.2 由 CPM 拉取，Serial 后端）：

- [x] Debug 与 Release 均配置成功，`cmake --build` 零警告（`/WX` 与 `-Werror` 均生效）
- [x] `ctest` 21/21 通过（Debug 与 Release 双配置）
- [x] 示例输出与解析解一致（`dot(y, y) = 144`）
- [x] 基准同时给出 numerix kernel、裸 Kokkos kernel 与手写循环的带宽，并带可校验的 checksum
- [x] `cmake --install` 后，独立使用方项目通过 `find_package(numerix)` 编译并运行成功
- [x] `add_subdirectory` 作为子项目时，tests/examples/benchmarks 默认关闭，numerix 目标可独立构建
- [x] `clang-format --dry-run --Werror` 全仓通过；`clangd --check` 对公共头文件零诊断

实测结论：Release 下 numerix 的 axpy 带宽与裸 Kokkos kernel 同量级（23.2 vs 24.8 GB/s），
手写循环略快（31.3 GB/s，Kokkos Serial 的逐元素调用开销属其序列后端特性，不是 numerix 的额外开销）。
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
- Windows 上 clang++（GNU 驱动）无法构建 KokkosKernels 5.2.2（它按 `CMAKE_CXX_SIMULATE_ID`
  判定 MSVC ABI 并注入 `/EHsc`），numerix 在配置阶段直接报错并给出替代方案；
  若上游修好该判定，可去掉 `cmake/Dependencies.cmake` 里的这道前置检查。
- Kokkos 默认开启 Serial；OpenMP/CUDA 后端需在对应环境下复验（`Kokkos_ENABLE_OPENMP` 等）。
- 复数标量的求解器测试待阶段 2 补充。
- 格式检查依赖 `uv tool run --from clang-format clang-format`，本机未安装独立 clang-format 二进制。
