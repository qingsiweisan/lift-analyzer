# Lift Analyzer - 力量举三大项动作分析工具

基于 MediaPipe 姿态估计的力量举动作分析 CLI 工具，支持硬拉、深蹲、卧推的视频分析。

## 功能

- **三大项分析** — 深蹲 (squat)、硬拉 (deadlift)、卧推 (bench press)
- **自动 Rep 计数** — 通过杠铃轨迹（手腕关键点追踪）自动识别动作次数
- **逐 Rep 评分** — 每个 rep 独立计算角度指标，给出 A/B/C 评级
- **角度分析** — 膝关节、髋关节、肘关节、脊柱前倾等关键角度
- **疲劳检测** — 对比首尾 rep 的动作质量变化
- **可视化输出** — 骨骼标注视频、角度变化曲线图、关键帧截图
- **中英双语报告**

## 各运动检测指标

| 硬拉 | 深蹲 | 卧推 |
|------|------|------|
| 脊柱前倾角度 | 膝关节最小角度（蹲深） | 肘关节角度范围 |
| 髋关节 ROM | 膝盖是否内扣 | 肩关节角度（手肘外展） |
| 锁定是否完全 | 躯干前倾（Good morning squat） | 左右对称性 |
| 杠铃轨迹 / 漂移 | 髋关节 ROM | 背弓角度 |

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:qingsiweisan/lift-analyzer.git
cd lift-analyzer
```

### 2. 安装依赖

```bash
# 推荐使用虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

依赖列表：
- `mediapipe` — Google 姿态估计
- `opencv-python` — 视频处理
- `numpy` — 数值计算
- `matplotlib` — 图表生成

### 3. 运行分析

```bash
# 硬拉分析
python analyze.py video.mp4 -t deadlift

# 深蹲分析
python analyze.py video.mp4 -t squat

# 卧推分析
python analyze.py video.mp4 -t bench
```

## 命令行参数

```
python analyze.py <video_path> [options]

必选：
  video_path              视频文件路径

可选：
  -t, --type TYPE         运动类型（默认 deadlift）
  -o, --output DIR        输出目录（默认 视频同目录/<type>_analysis/）
  --no-video              不生成标注视频（加速处理）
  --no-chart              不生成角度变化图表
```

支持的类型名和缩写：

| 类型 | 缩写 | 中文 |
|------|------|------|
| `deadlift` | `dl` | `硬拉` |
| `squat` | `sq` | `深蹲` |
| `bench` | `bp` | `卧推` |

## 输出文件

每次分析在输出目录下生成：

```
<output_dir>/
├── annotated.mp4       # 带骨骼 + 角度标注的视频
├── report.txt          # 中英双语动作评估报告
├── frame_data.json     # 逐帧角度数据（可用于二次分析）
├── charts.png          # 角度随时间变化曲线图（含 rep 标记）
└── frames/             # 关键帧截图（0%/25%/50%/75%/100%）
    ├── frame_0000.jpg
    ├── frame_0540.jpg
    └── ...
```

## 示例输出

### 报告示例（硬拉）

```
============================================================
     DEADLIFT FORM ANALYSIS / 硬拉动作分析报告
============================================================
Duration: 36.3s | Frames: 2169
Reps detected / 检测到次数: 3
  (via barbell tracking / 通过杠铃轨迹检测)

--- PER-REP BREAKDOWN / 逐次分析 ---
  Rep      Time  Bar Travel   Hip ROM   Max Back   Lockout   Grade
  # 1     5.3s      17.0%     104 deg       88 deg     165 deg       B
  # 2    19.7s      17.4%     106 deg       89 deg     165 deg       B
  # 3    35.4s      15.8%      96 deg       88 deg     160 deg       C
```

### 评级标准

| 评级 | 条件（满足越多越高） |
|------|---------------------|
| **A** | 髋ROM>=60° + 最大前倾<=50° + 锁定>=160° |
| **B** | 满足其中 2 项 |
| **C** | 满足其中 0-1 项 |

## 拍摄建议

为了获得最佳分析效果：

- **最佳角度**：纯侧面（90°），相机与髋同高
- **可接受**：侧后方 45° 也能用，但角度计算精度会下降
- **确保全身入镜**：从头到脚都在画面内
- **背景简洁**：避免多人同时在画面中
- **光线充足**：避免逆光和强阴影

## 项目结构

```
lift-analyzer/
├── analyze.py              # CLI 入口
├── requirements.txt        # Python 依赖
├── core/                   # 通用模块
│   ├── angles.py           #   角度计算工具
│   ├── pose.py             #   MediaPipe 姿态提取封装
│   ├── video.py            #   视频读写、关键帧采样
│   ├── chart.py            #   matplotlib 角度曲线图
│   └── reps.py             #   峰值检测 / Rep 计数
├── exercises/              # 运动分析器
│   ├── base.py             #   分析器基类
│   ├── deadlift.py         #   硬拉（含杠铃轨迹追踪）
│   ├── squat.py            #   深蹲（含膝盖内扣检测）
│   └── bench.py            #   卧推（含左右对称性检测）
└── .gitignore
```

## 技术架构

```
视频输入
  │
  ▼
MediaPipe BlazePose (33个关键点)
  │
  ├── 自动选择可见侧（左/右）
  ├── 提取双侧关键点（对称性分析）
  │
  ▼
运动分析器 (deadlift/squat/bench)
  │
  ├── 逐帧角度计算
  ├── 杠铃追踪（手腕 Y 坐标）
  ├── Rep 检测（峰值/谷值检测）
  ├── 逐 Rep 评分
  │
  ▼
输出：标注视频 + 报告 + 图表 + JSON + 关键帧
```

## 已知局限

- **2D 姿态估计** — 基于单目 2D 投影，拍摄角度会影响角度计算精度
- **杠铃追踪** — 用手腕关键点近似，非直接识别杠铃
- **单人场景** — 画面中如有多人可能影响检测
- **脊柱分析** — 只能用肩-髋连线估算前倾，无法检测脊柱弯曲弧度

## 环境要求

- Python 3.9+
- 无需 GPU（CPU 即可运行，GPU 可加速）
- Windows / Linux / macOS 均可
