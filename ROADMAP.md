# Roadmap / 开发计划

## 当前版本 v0.1 — 已完成

- [x] MediaPipe BlazePose 2D 姿态估计
- [x] 硬拉分析（角度 + 杠铃轨迹 + rep 计数）
- [x] 深蹲分析（蹲深 + 膝盖内扣 + 前倾检测）
- [x] 卧推分析（肘关节 ROM + 手肘外展 + 对称性）
- [x] CLI 命令行工具
- [x] 标注视频 / 角度曲线图 / 中英双语报告
- [x] 逐 Rep 评分 (A/B/C)
- [x] 疲劳检测（首尾 rep 对比）

## v0.2 — 短期目标

- [ ] 3D 姿态估计集成（WHAM），解决拍摄角度依赖问题
  - WHAM: https://github.com/yohanshin/WHAM
  - 需要 CUDA GPU（RTX 3060+ / 8GB+ VRAM）
  - 预计角度精度从 ~15-20° 误差降至 ~5°
- [ ] YOLOv8 杠铃检测，替代手腕近似
- [ ] 杠铃轨迹图（Bar Path Visualization）—— 每个 rep 的 X-Y 轨迹
- [ ] 速度/加速度分析 —— 基于关键点的时间微分，输出发力曲线

## v0.3 — 中期目标

- [ ] Android App（手机端录制 + PC 端分析）
  - 手机端：MediaPipe BlazePose Lite 实时预览（835 可跑 ~31fps）
  - PC 端：WHAM 3D 分析
- [ ] Web 界面（Gradio/Streamlit）—— 拖入视频即可分析
- [ ] 多组对比 —— 同一动作不同日期的进步追踪
- [ ] 训练日志集成 —— 结合重量/组数记录

## v1.0 — 长期目标

- [ ] OpenCap Monocular 集成 —— 生物力学级精度（关节力矩、肌肉激活）
- [ ] 多运动扩展 —— 推举 (OHP)、划船 (Row)、高翻 (Clean) 等
- [ ] 个人动作模型 —— 基于历史数据建立个人基准，检测异常偏差
- [ ] 实时语音反馈 —— 练的时候直接告诉你"再蹲深一点"

## 技术调研笔记

### 3D 姿态估计方案对比

| 方案 | 精度 | 速度 (RTX 4090) | VRAM | 手机端 |
|------|------|-----------------|------|--------|
| MediaPipe BlazePose (当前) | ~15-20° | 实时 | 无需GPU | 31fps (835) |
| WHAM | ~5° | ~48s/45s视频 | ~10GB | 不可能 |
| OpenCap Monocular | ~4.8° | 更慢 | ~10GB | 不可能 |
| MotionBERT | ~5-8° | 中等 | ~8GB | 不可能 |

### 杠铃检测方案

| 方案 | 说明 |
|------|------|
| 手腕近似 (当前) | 简单有效，但不精确 |
| YOLOv8 目标检测 | 需要训练/找预训练模型 |
| 颜色检测 (HSV) | 对特定杠铃片颜色有效（如 ELEIKO 黄色） |
| Roboflow 预训练模型 | https://universe.roboflow.com/search?q=barbell |

### 关键参考项目

- WHAM: https://github.com/yohanshin/WHAM (CVPR 2024)
- OpenCap: https://www.opencap.ai/ (斯坦福)
- Pose2Sim: https://github.com/perfanalytics/pose2sim
- MotionBERT: https://github.com/Walter0807/MotionBERT (ICCV 2023)
- pose-estimation-for-powerlifting: https://github.com/03y/pose-estimation-for-powerlifting
