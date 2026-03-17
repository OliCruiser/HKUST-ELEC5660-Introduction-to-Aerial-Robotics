<img src="https://wpcos-1300629776.cos.ap-chengdu.myqcloud.com/picgo/Gemini_Generated_Image_brfxjrbrfxjrbrfx.png" alt="Gemini_Generated_Image_brfxjrbrfxjrbrfx" style="zoom: 25%" />

## 香港科技大学 ELEC5660：空中机器人导论

ELEC5660 是香港科技大学的一门研究生课程，旨在系统介绍空中机器人相关内容。本课程的目标是让学生接触相关的数学基础与算法，并训练他们开发适用于空中机器人系统的实时软件模块。课程涵盖的主题包括刚体动力学、系统建模、控制、轨迹规划、传感器融合，以及基于视觉的状态估计。学生将完成一系列项目，这些项目最终会组合成一个能够基于视觉进行室内自主导航的空中机器人系统。

**授课教师：** [Shaojie SHEN](https://ece.hkust.edu.hk/eeshaojie)

**助教：** [Yang Xu](https://jason-xy.cn/self/) (yxuew@connect.ust.hk), Pusen Gao (pgaoak@connect.ust.hk)

**实验室：** [HKUST Aerial Robotics Group](https://uav.ust.hk)

---

### 文件结构

* `course_node`：课程笔记
* `assignments`：作业代码
* `lab`：实验笔记
* `workspace_template`：用于搭建云端工作区的 Terraform 模板

### 作业与实验简要说明

| 名称 | 描述 | 演示 |
|------|------|------|
| proj1phase1 | 实现一个用于四旋翼轨迹跟踪的基础控制器 | ![p1p1](fig/p1p1.gif) |
| proj1phase2 | 实现基于最小 jerk/snap 优化的轨迹生成 | ![p1p2](fig/p1p2.gif) |
| proj1phase3 | 实现 A* 路径规划，并与轨迹生成和控制模块集成 | ![p1p3](fig/p1p3.gif) |
| lab1 | 组装并以手动模式飞行无人机，同时为后续实验准备软硬件环境 | ![IMG_6944](fig/IMG_6944.jpg) |
| proj1phase4_lab2 | 在 OptiTrack 动作捕捉系统辅助下，以自主控制模式飞行无人机，并分析飞行数据 | ![lab2](fig/lab2.gif) |
| proj2phase1 | 实现用于视觉状态估计的 Perspective-n-Point (PnP) 算法 | ![pnp](fig/pnp.gif) |
| proj2phase2 | 实现用于 6 自由度位姿估计的双目视觉里程计 | ![stereo_vo](fig/stereo_vo.gif) |
| proj3phase1 | 实现扩展卡尔曼滤波器（EKF），融合 IMU 与视觉里程计数据 | ![p3p1](fig/p3p1.gif) |
| proj3phase2 | 实现增强型 EKF，融合 IMU、视觉里程计和基于标签的位姿估计 | ![p3p2](fig/p3p2.gif) |
| proj3phase3_lab3 | 将整个系统部署到机载平台，实现轨迹跟踪，或在不依赖 OptiTrack 动作捕捉系统的情况下完成自主飞行 | ![p3p3](fig/p3p3.gif) |

### 模拟器（可选）

为了方便算法开发与测试，我们提供了一个基于 NVIDIA Isaac Sim 的模拟器。该模拟器支持较为真实的物理仿真、传感器仿真（IMU、双目相机）以及 ROS 接口，便于与你的代码集成。你可以在 `lab/simulator` 目录中找到模拟器代码和使用说明。

![sim_demo](fig/sim_demo.gif)

### 联系方式

如有问题或建议，请联系助教。
