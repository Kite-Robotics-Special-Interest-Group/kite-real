# kite-real

> 真机 VLA 研究：数据采集、微调，以及**用输入扰动预测真机失败**。

仿真线已收尾（结论见 [`kite-bench/docs/handoff.md`](https://github.com/Kite-Robotics-Special-Interest-Group/kite-bench/blob/main/docs/handoff.md)），这里开始真机阶段。仓库只放研究代码与协议，不放大文件。

## 研究问题

**能不能在真机跑之前，用"输入扰动下的动作漂移"预判它会在什么条件下失败？**

这个问题来自仿真里两个可复现的现象：

1. 外观一变（只加传送带/料筐，任务几何不变），成功率从 8/8 掉到 **0/8**
2. 同一模型在模糊 3 px 时动作漂移 **70 mm**、10% 面积遮挡时夹爪判断翻转 **31%**

如果"扰动漂移"与"真机成功率"之间存在稳定映射，就可以把评测从昂贵的真机试验前移到离线计算。**可证伪**：若漂移与失败相关性弱（Spearman < 0.3），假设不成立。

## 三个仓库的边界

| 仓库 | 管什么 |
| --- | --- |
| `Kite` | 端侧模型与运行时（板端） |
| `kite-bench` | 仿真与离线评测工具（已结项，工具仍可用） |
| `kite-showcase` | Unity 场景、视频渲染、素材规范 |
| `kite-weekly` | 周报归档 |
| **`kite-real`** | 真机数据、微调实验、鲁棒性研究 |

## 目录

```
src/kite_real/
  schema.py       # episode 格式（与 kite-bench 一致）+ 校验
  record.py       # 录制写入器（真机采集用）
  perturb_eval.py # 真机图像扰动评测（复用 kite-bench 的扰动与指标）
docs/
  design.md       # 假设、指标、实验设计、可证伪条件
  protocol.md     # 真机采集与评测协议（安全 / 标定 / 频率 / 日志）
  plan.md         # 团队计划表：分工、里程碑、依赖
  arm-modeling.md # 机械臂建模所需数据（测量、标定、交付物）
```

## 团队分工

见 [`docs/plan.md`](docs/plan.md)：5 人分工、三周里程碑、依赖关系。机械臂建模的数据清单见 [`docs/arm-modeling.md`](docs/arm-modeling.md)。



## 快速开始

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## 许可

MIT，见 [`LICENSE`](LICENSE)。

