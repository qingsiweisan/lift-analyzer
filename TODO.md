# Lift Analyzer — 开发计划 / TODO

> v0.1 已完成（MediaPipe 2D）。下一步优先集成 WHAM 3D 姿态估计。
> 开发机器：RTX 5070 Ti (16GB VRAM)

---

## v0.2 — WHAM 3D 集成 + 分析增强

### Milestone 1.1: WHAM 环境搭建
> ⚠️ 需要在 5070 Ti 机器上操作（需要 CUDA GPU）

- [ ] 检查 CUDA / cuDNN / conda 环境
  ```bash
  nvidia-smi
  conda --version
  ```
- [ ] clone WHAM 仓库
  ```bash
  git clone --recursive https://github.com/yohanshin/WHAM.git
  cd WHAM
  ```
- [ ] 创建 conda 环境
  ```bash
  conda create -n wham python=3.9
  conda activate wham
  conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
  ```
- [ ] 安装 ViTPose / DPVO 等依赖（按 WHAM/docs/INSTALL.md）
- [ ] 注册并下载 SMPL 模型
  - 注册地址：https://smpl.is.tue.mpg.de
  - 运行 `bash fetch_demo_data.sh`
- [ ] 跑通 demo
  ```bash
  python demo.py --video examples/IMG_9732.mov --visualize
  ```
- **完成标志**：WHAM demo 视频输出 3D 人体网格

### Milestone 1.2: WHAM 适配层
> 新增 `core/pose_3d.py`，修改 `analyze.py` 和 `exercises/base.py`

- [ ] 封装 WHAM Python API（输入视频路径 → 输出逐帧 3D 关节坐标）
- [ ] SMPL 24 joints → 我们的关键点名映射（shoulder/hip/knee/ankle/wrist/elbow）
- [ ] `analyze.py` 新增 `--backend` 参数：`mediapipe`（默认）/ `wham`
- [ ] `base.py` 的 `run()` 根据 backend 选择姿态提取方式
- **完成标志**：`python analyze.py video.mp4 -t deadlift --backend wham` 出报告

### Milestone 1.3: 3D 角度计算
> 修改 `core/angles.py`

- [ ] `calculate_angle_3d(a, b, c)` — 三维空间向量夹角
- [ ] `spine_inclination_3d(shoulder, hip)` — 真实 3D 前倾角度
- [ ] 分析器自动检测输入维度（2D tuple vs 3D tuple），选择对应函数
- **完成标志**：3D 模式下角度不再受拍摄角度影响

### Milestone 1.4: YOLOv8 杠铃检测
> 新增 `core/barbell.py`

- [ ] 调研 Roboflow 预训练 barbell 模型（https://universe.roboflow.com/search?q=barbell）
- [ ] 集成 ultralytics YOLOv8（`pip install ultralytics`）
- [ ] 杠铃 bounding box → 中心点坐标
- [ ] 替换手腕近似做 rep 计数和轨迹分析
- [ ] fallback：检测不到时退回手腕近似
- **完成标志**：杠铃追踪精度提升，rep 计数更准

### Milestone 1.5: Bar Path 可视化 + 速度曲线
> 修改 `core/chart.py`

- [ ] Bar Path 图：每个 rep 的 X-Y 轨迹叠加
- [ ] 速度曲线：杠铃 Y 坐标时间微分 → concentric/eccentric 速度
- [ ] 加速度 / 发力峰值标注
- **完成标志**：输出新增 `bar_path.png` 和 `velocity.png`

### v0.2 发布检查清单
- [ ] `--backend wham` 跑通三大项
- [ ] 3D vs 2D 角度对比验证
- [ ] 杠铃检测 + bar path 图
- [ ] 速度/发力曲线
- [ ] 更新 README

---

## v0.3 — 用户界面 + 数据管理

### Milestone 2.1: Web UI（Gradio）
- [ ] 上传视频 → 选择运动类型 → 展示报告/图表/标注视频
- [ ] 支持 mediapipe / wham 后端切换
- [ ] 一键启动：`python app.py`

### Milestone 2.2: 训练日志 + 多组对比
- [ ] 本地 SQLite 存储分析历史（日期、运动、重量、组数、评分）
- [ ] 同一动作不同日期的角度趋势对比
- [ ] 进步追踪：ROM 改善、前倾改善、rep 一致性
- [ ] `python analyze.py video.mp4 -t deadlift --weight 140 --save`

### Milestone 2.3: 手机端方案（选一）
- [ ] 方案 A：watchdog 监控文件夹，手机传视频自动分析（最简单）
- [ ] 方案 B：Gradio 部署局域网，手机浏览器上传（推荐）
- [ ] 方案 C：原生 Android App（复杂，MediaPipe 实时预览 + 上传 PC）

---

## v1.0 — 高级分析

### Milestone 3.1: OpenCap Monocular 集成
- [ ] 替代 WHAM → 生物力学级关节运动学
- [ ] 关节力矩估计 + 肌肉激活估计

### Milestone 3.2: 更多运动
- [ ] 推举 (OHP) — 肩关节 ROM + 腰椎过伸
- [ ] 划船 (Row) — 躯干角度 + 肘关节 ROM
- [ ] 高翻 (Clean) — 三拉阶段识别
- [ ] 每种运动一个 `exercises/<name>.py`

### Milestone 3.3: 个性化模型
- [ ] 基于历史数据建立个人基准角度
- [ ] 异常偏差检测（"这个 rep 膝盖内扣比你平时大 15°"）

### Milestone 3.4: 实时反馈
- [ ] MediaPipe 实时 + TTS 语音提示（"再蹲深一点"）

---

## 执行流程图

```
当前 ──→ Milestone 1.1 (WHAM 环境搭建，在 5070Ti 机器上)
           │
           ▼
         Milestone 1.2 (WHAM 适配层)
           │
           ▼
         Milestone 1.3 (3D 角度计算)
           │
           ├── 可并行 ──→ Milestone 1.4 (YOLOv8 杠铃)
           │              Milestone 1.5 (Bar Path + 速度)
           ▼
         ═══ v0.2 发布 ═══
           │
           ▼
         Milestone 2.1 (Gradio Web UI)
           │
           ▼
         Milestone 2.2 + 2.3 (日志 + 手机)
           │
           ▼
         ═══ v0.3 发布 ═══
           │
           ▼
         Phase 3 (v1.0，按需推进)
```

## 技术参考

| 项目 | 链接 | 用途 |
|------|------|------|
| WHAM | https://github.com/yohanshin/WHAM | 3D 姿态估计 (CVPR 2024) |
| OpenCap | https://www.opencap.ai/ | 生物力学分析 (斯坦福) |
| MotionBERT | https://github.com/Walter0807/MotionBERT | 3D lifting (ICCV 2023) |
| Pose2Sim | https://github.com/perfanalytics/pose2sim | 多摄像头 3D |
| Roboflow Barbell | https://universe.roboflow.com/search?q=barbell | 杠铃检测模型 |
