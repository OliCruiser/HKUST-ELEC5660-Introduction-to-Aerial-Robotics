# 项目 1 第二阶段：轨迹生成

布置时间：2026 年 3 月 3 日

截止时间：2026 年 3 月 13 日

## 概述

在第一阶段中，你已经实现了一个用于轨迹跟踪的控制器。在第二阶段中，你将重点实现 **轨迹生成器（Trajectory Generator）**。一个设计良好的轨迹生成器能够让四旋翼以更激进且更精确的方式飞行。

## 目标

1. 实现轨迹生成算法，用于连接一系列航点。
2. 保证轨迹满足平滑性要求（位置、速度、加速度连续）。
3. 实现基于优化的轨迹生成方法（Minimum Jerk 或 Minimum Snap）。

| ![image-20260128220915420](https://wpcos-1300629776.cos.ap-chengdu.myqcloud.com/picgo/image-20260128220915420.png) | ![image-20260128220950766](https://wpcos-1300629776.cos.ap-chengdu.myqcloud.com/picgo/image-20260128220950766.png) | ![image-20260128221017859](https://wpcos-1300629776.cos.ap-chengdu.myqcloud.com/picgo/image-20260128221017859.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

## 你的任务

完成 `sim/trajectory_generator.py` 中的 `TrajectoryGenerator` 类。你需要实现以下函数：

1. `_generate_smooth_only`：一种朴素方法，保证几何连续性（$C^2$），但不最小化任何代价函数。（使用五次样条）
2. `_generate_minimum_jerk`：最小化 jerk（三阶导数）的平方积分。（使用六阶多项式）
3. `_generate_minimum_snap`：最小化 snap（四阶导数）的平方积分。（使用八阶多项式）

## 运行仿真

1. 运行网页界面：
   ```bash
   python app.py
   ```

2. 在浏览器中打开 `http://localhost:8080`

3. 选择一条 “Path”（航点集合）以及一种 “Optimization” 方法：
   - *Smooth Only*
   - *Minimum Jerk*
   - *Minimum Snap*

4. 点击 “Run Simulation”。

## 评估标准

1. **航点连接**：轨迹必须经过所有给定航点。
2. **平滑性**：四旋翼飞行应当平滑，位置、速度和加速度不能出现不连续。
3. **跟踪误差**：在控制器相同的情况下，更优的轨迹（例如 Min Snap）应当带来更低的跟踪误差。

## Tips

- 已经为你提供了辅助函数 `_solve_equality_qp(H, Aeq, beq)`，它可以求解标准二次规划问题：
  $$ \min \frac{1}{2} x^T H x \quad \\ \text{s.t.} \quad A_{eq} x = b_{eq} $$
  你可以利用它来求解多项式系数。
- 复习 Lecture 4 中关于轨迹生成的课件内容。
- 将几何约束（航点、连续性）映射为线性约束 $A_{eq} x = b_{eq}$。
- 将代价函数中的积分项映射为 Hessian 矩阵 $H$。

## 已提供的代码结构

- `app.py`：用于运行仿真的网页界面
- `sim/trajectory_generator.py`：**你需要在这里实现代码**
- `sim/trajectories.py`：预定义路径（航点）以及轨迹辅助工具
- `sim/controller.py`：第一阶段中的控制器（用于跟踪生成出的轨迹）
- `sim/simulator.py`：仿真引擎
- `sim/dynamics.py`：四旋翼动力学模型
- `sim/model.py`：四旋翼参数
- `sim/math_utils.py`：旋转相关辅助函数
- `sim/visualization.py`：绘图工具

## 提交要求

提交你完成的代码以及一份简短报告。

代码应包含运行你的轨迹生成器仿真所需的全部文件。

报告最多 2 页，应包含以下内容：

- 仿真器生成的图像（至少在一条路径上对比 Smooth Only / Min Jerk / Min Snap）
- 关于轨迹/跟踪效果的统计数据（例如位置 RMS 误差、速度/加速度/jerk 峰值）
- 对结果的分析（例如参数选择、时间分配研究）
- 任何其他我们需要了解的内容

请将所有内容打包为一个名为 `proj1phase2_yourname.zip` 的压缩文件，并提交到 Canvas。

请注明你在完成本次作业时参考的论文、GitHub 仓库或其他资源。请遵守[学术诚信](https://registry.hkust.edu.hk/resource-library/academic-integrity)要求，本课程不容忍抄袭行为。

## 延期提交政策

允许在截止日期后 7 天内迟交，但每天将扣除该项总分的 5%。
