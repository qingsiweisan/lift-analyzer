# Lift Analyzer — 开发计划 / TODO

> v0.1 已完成（MediaPipe 2D）。v0.2 WHAM 3D 集成基本完成。
> 开发机器：RTX 5070 Ti (16GB VRAM)，WHAM 运行在 WSL2 Ubuntu

---

## v0.2 — WHAM 3D 集成 + 分析增强

### Milestone 1.1: WHAM 环境搭建 ✅
> WSL2 Ubuntu + conda env "wham" + CUDA 12.8

- [x] CUDA / cuDNN / conda 环境（WSL2 Ubuntu）
- [x] clone WHAM 仓库（/root/WHAM）
- [x] conda 环境 wham (Python 3.9, PyTorch 1.11, CUDA 11.3)
- [x] ViTPose / DPVO 等依赖
- [x] SMPL 模型下载
- [x] demo 跑通
- **完成**：WHAM demo 输出 3D 人体网格 ✅

### Milestone 1.2: WHAM 适配层 ✅
> `core/pose_3d.py` — Windows→WSL 桥接，自动调用 WHAM

- [x] 封装 WHAM API（Windows Python → WSL conda → demo.py → pkl → joints）
- [x] SMPL 24 joints → 关键点映射（通过 J_regressor 从 6890 vertices 提取）
- [x] `analyze.py` 新增 `--backend wham`
- [x] `base.py` 的 `run()` 根据 backend 选择姿态提取方式
- [x] WHAM 输出缓存（跳过重复推理）
- **完成**：`python analyze.py video.mp4 -t deadlift --backend wham` 出报告 ✅

### Milestone 1.3: 3D 角度计算 ✅
> `core/angles.py` — 3D 向量夹角 + 自动 2D/3D 切换

- [x] `calculate_angle_3d(a, b, c)` — 三维空间向量夹角
- [x] `spine_inclination_3d(shoulder, hip)` — 真实 3D 前倾角度（SMPL Y-up 修正）
- [x] `auto_angle()` / `auto_spine_inclination()` 自动检测维度
- [x] deadlift.py 适配 3D 模式（bar tracking、rep detection、report）
- **完成**：3D 模式下角度不受拍摄角度影响 ✅

### Milestone 1.4: YOLOv8 杠铃检测 (部分完成)
> `core/barbell.py` — 代码已写好，缺训练好的模型

- [x] 集成 ultralytics YOLOv8（已安装）
- [x] `core/barbell.py` BarbellTracker 类（检测 + fallback 逻辑）
- [x] `base.py` 集成 `--barbell-model` 参数
- [ ] 标注杠铃数据集 + 训练自定义模型（Roboflow 无现成可下载模型）
- [ ] 替换手腕近似做 rep 计数和轨迹分析
- **状态**：代码就绪，等模型训练。3D 模式下手腕近似已足够准确

### Milestone 1.5: Bar Path 可视化 + 速度曲线 ✅
> `core/chart.py` — 已在 v0.1 实现，v0.2 适配 3D 坐标

- [x] Bar Path 图：每个 rep 的 X-Y 轨迹叠加
- [x] 速度曲线：杠铃 Y 坐标时间微分 → concentric/eccentric 速度
- [x] 加速度 / 发力峰值标注
- [x] 3D 模式下使用米为单位
- **完成**：输出 `bar_path.png` 和 `velocity.png` ✅

### v0.2 发布检查清单
- [x] `--backend wham` 跑通硬拉（deadlift）
- [x] `--backend wham` 跑通深蹲（squat）和卧推（bench）
- [x] 3D vs 2D 角度对比验证（3D 角度更合理，rep 检测更准）
- [x] 杠铃检测代码就绪（`--barbell-model`），待训练模型
- [x] 速度/发力曲线
- [x] 更新 README

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
