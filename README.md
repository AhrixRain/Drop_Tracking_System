# Drop_Tracking_System

密立根油滴实验（Physics 121W）的自动追踪与分析系统：从视频帧自动追踪单个油滴的运动，经高度/速度/加速度分析，最终按 UCI 讲义方法计算油滴所带电荷。

## 流水线

```
视频 / 帧目录
   │
   ├─ detect.py          油滴检测与追踪 → drop_pixel_coords.csv（每帧坐标）
   │
   ├─ diff_bg_frame.py   调试可视化：原始/背景/差分/二值 四联对比图
   │
   ├─ plot.py
   │   ├─ --mode trajectory       轨迹散点图（按时间着色）
   │   └─ --mode height_velocity  高度/速度/加速度图 + height_velocity CSV
   │
   └─ calc_uci.py        电荷计算 → *_calculated_results_uci.csv（含 n_e）
```

## 安装

```bash
pip install -r requirements.txt
```

## 用法

### 1. 检测与追踪（输入视频或帧目录）

```bash
python detect.py --input frames_sample18/ --output drop_pixel_coords.csv
python detect.py --input movie.mp4 --output drop_pixel_coords.csv --fps 60
```

可选参数：`--bg-samples`（背景采样帧数）、`--threshold`（灵敏度）、`--min-area/--max-area/--circularity`（blob 过滤）、`--gate-radius/--max-missed/--ema-alpha`（跟踪）、`--reinit-rule {largest_blob,highest_conf,nearest_center}`。

### 2. 调试可视化（可选）

```bash
python diff_bg_frame.py --input frames_sample18/ --output-dir diff_bg_frames/
```

### 3. 绘图与运动学分析

```bash
# 轨迹散点图
python plot.py --input drop_pixel_coords.csv --mode trajectory --confidence 0.3
# 高度/速度/加速度（生成 height_velocity CSV）
python plot.py --input drop_pixel_coords.csv --mode height_velocity
```

### 4. 电荷计算

```bash
python calc_uci.py --input height_velocity_sample18.csv        # 直接读速度
python calc_uci.py --input drop_pixel_coords.csv --method free_fall
```

更多选项见 `python calc_uci.py --help`（电压、极板间距、温度、气压、油密度、标定像素等均可覆盖）。

## 标定说明

所有脚本的标定参数默认值集中在 `config.py`，也可用 CLI 覆盖：

- `--ceiling-y-px / --floor-y-px`：图像中标定高度范围对应的顶部/底部 y 像素。
- `--plate-spacing`：极板间距（米），兼作像素标定高度跨度。

**务必按你的实际实验装置设置这些值**，否则高度与电荷结果会失真。
