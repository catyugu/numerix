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

## 阶段 1：动态代数补齐（已完成）

- `MultiVector`：块 Krylov、多右端、POD/ROM 的基础容器
- 稀疏算子包装 `SparseOperator`（基于 `KokkosSparse::CrsMatrix`，不自行发明 CSR）
- 向量空间/内积语义的统一约定（实数与复数）

验收（2026-10-14，Linux / GCC 13.3.0，Kokkos / Kokkos Kernels 5.2.2，Serial）：

- [x] `MultiVector` 的列内积与逐列 `Dot(exec, vector.View(), ...)` 一致，支持跨步视图与复数共轭。
- [x] move-only、只读 View、Clone / DeepCopy 独立存储与空尺寸语义通过验证。
- [x] `SparseOperator` 的矩形乘法、空行、零矩阵与手算一致，复数乘法不错误共轭。
- [x] 线性性、Hermitian 对称性、正定性、融合 Apply 和块乘法与逐列结果一致。
- [x] CSR offset 接受原生有符号整数与 `std::size_t`，ordinal / offset 独立。
- [x] Debug 与 Release 零编译警告（`-Wall -Wextra -Wpedantic -Werror`），CTest 均 34/34。
- [x] Debug 与 Release 的安装迁移消费和父项目嵌入均通过，新 API 纳入消费者测试。
- [x] 新头文件独立包含可编译，修改的 C++ 文件通过 clang-format 检查。

TDD 记录（最终新增 13 项测试）：先添加数学性质与数据契约测试，缺失头文件导致编译失败；
最小实现后首次 CTest 为 32/33，空列 dot 触发 Kokkos 多向量 reduction 的数值异常。
保留失败测试，以空列直接返回、空行 deep_copy 零值修复。
随后新增原生 `int` offset 测试，确认 unsigned-only 约束编译失败，再放宽为非 bool 整数。
全部通过后，将稀疏 Apply 重构为显式 `SPMV_NATIVE` 的栈上 handle，重跑全套验证。
跨平台 CI 又发现 Kokkos Kernels 5.2.2 的 SpMV handle 在 `throw` 后保留了不可达 return，
导致 MSVC Debug 的 C4702 被 `/WX` 拒绝。对固定版本添加仅删除该语句的依赖补丁，
保持警告报错策略，复验 Debug / Release。

复现：

```bash
cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DNUMERIX_WARNINGS_AS_ERRORS=ON
cmake --build build-debug --parallel 2
ctest --test-dir build-debug --output-on-failure
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DNUMERIX_WARNINGS_AS_ERRORS=ON
cmake --build build-release --parallel 2
ctest --test-dir build-release --output-on-failure
```

阶段 1 泛型约束修复（2026-10-14）：

- 补齐 Scalar 的表达式与返回类型约束，owning 容器拒绝 cv/ref 元素类型；标量辅助函数返回值而非引用。
- 动态数值内核检查 Kokkos ArithTraits 注册；存储自定义标量不额外要求后端算术支持。
- Axpy 按真实后端系数类型和表达式能力约束混合组合，定长向量允许可转换系数，SparseOperator 接受原生只读 CSR。
- TDD：先新增编译期拒绝测试和数学正向测试，确认原实现失败；再修复，最后执行完整回归。
  新增 5 项运行期 / constexpr 数学测试及编译期拒绝断言，Debug / Release 均 39/39 通过。
  算子的成员 Apply 入口保持现有约定。具体边界见 DESIGN.md §5.1 / §5.4。

下一步：阶段 2 首先按 TDD 将求解器代数契约与 CG 一起落地（见 DESIGN.md §5.9），
再扩展预条件子和其他迭代法。现有 View 运算保留本地语义；本次只确定文档约定，
没有新增求解器、代数操作对象或 MPI 实现。

## 阶段 2：线性求解器

### 第一步：代数操作接口与 CG 同时落地

1. 先写失败测试：已知谱实数 SPD、复数 Hermitian 正定系统，以及能区分本地归约和完整向量结果的
   受控分区测试代数实现。验证 CG 的实际步长、真实残差与退出状态，不能只检查接口调用次数。
2. 实现受 concept 约束、静态绑定的最小代数操作接口及本地实现；调用方显式绑定 exec。
   工作向量创建、复制、更新、内积、范数与向量兼容性经该接口完成。接口只收录 CG 的真实需要，
   与 CG 一起交付，不提前添加通用向量空间框架或 MPI 空壳。
3. 实现 `SolveResult Solve(algebra, const A& op, const X& b, X& x)` 的入口语义、CG、Identity 预条件子
   与收敛结果。这里是接口草案，完整 C++ 签名和算法配置在实施时由测试确定。
   求解器不得写死 Vector / View 存储，不得绕过代数对象直接调用本地 Dot / Norm2 或自行调用 MPI。
4. 在测试保持通过的前提下重构，验证显式执行流、工作空间所有权、归约语义和收敛判断。
   本地与受控分区代数实现通过同一 CG 算法；实际 MPI 正确性仍由阶段 4 多进程测试确认。

本地代数内部复用现有 View 内核；算子 Apply 保留显式 exec，并与代数对象绑定的执行流一致。
分布式 owned / ghost、全局归约、空间兼容性和操作完成契约以 DESIGN.md §5.9 为准。
进入其他迭代法前必须先完成上述步骤。

### 后续交付

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
- 实现阶段 2 已确定的代数契约：全局内积与范数、按分布创建工作向量、owned / ghost 语义和分布兼容性
- 分布式算子组合本地 SparseOperator / matrix-free 算子与通信计划，明确 domain/range 分布
- 区分 halo 刷新与反向贡献累加，明确通信请求、缓冲区生命周期及 Kokkos 执行流同步
- 可选依赖不得进入核心头文件；通用求解器不依赖具体 MPI 类型

验收：不均匀分区与空 owned 分区、ghost 不重复计数、分布不兼容拒绝、halo 刷新与反向累加；
单进程与多进程结果一致，所有进程的收敛状态和退出一致；adapter 关闭时核心库仍可独立构建。

## 阶段 5：诊断与性能

- `Timer`、`Profiler`、`ResidualHistory`、`ConvergenceMonitor`
- 基准扩展到稀疏矩阵-向量乘、点积、流水线的内存带宽与强/弱扩展性

验收：基准给出可复现的带宽与扩展性数据；诊断开销在热路径中可关闭。

## 待办与风险

- GCC 13.3.0 / Serial 的 Debug 与 Release 已复验；OpenMP 与设备后端仍需对应环境验证。
- Windows 上 clang++（GNU 驱动）无法构建 KokkosKernels 5.2.2（它按 `CMAKE_CXX_SIMULATE_ID`
  判定 MSVC ABI 并注入 `/EHsc`），numerix 在配置阶段直接报错并给出替代方案；
  若上游修好该判定，可去掉 `cmake/Dependencies.cmake` 里的这道前置检查。
- Kokkos 默认开启 Serial；OpenMP/CUDA 后端需在对应环境下复验（`Kokkos_ENABLE_OPENMP` 等）。
- 复数标量的求解器测试待阶段 2 补充。
- 格式检查依赖 `uv tool run --from clang-format clang-format`，本机未安装独立 clang-format 二进制。
