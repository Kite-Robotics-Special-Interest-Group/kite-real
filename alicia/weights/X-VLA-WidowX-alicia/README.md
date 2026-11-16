# X-VLA Alicia-D soft prompt (domain 19)

在 `2toINF/X-VLA-WidowX` 原版权重上，为 Alicia-D v5.6（50 mm 夹爪）训练的
单臂 domain 权重。

## 文件

- `softprompt_domain19.pt`：当前 interleaved rot6d 版本（2 epoch，约 0.5 MB）；
- `softprompt_domain19_columns.pt`：旧的 column-major rot6d 备份，仅用于对照。

`softprompt_domain19.pt` 只含 domain 19 的：

- soft prompt（32 × 1024）
- action encoder 的权重/偏置行
- action decoder 的权重/偏置行

权重文件不进 Git（`*.pt` 忽略），通过网盘分发。

## 关键修正

2026-11-15 发现官方 X-VLA 的 rot6d 是**行交错**
`[b1x,b2x,b1y,b2y,b1z,b2z]`，不是 `[b1x,b1y,b1z,b2x,b2y,b2z]`。
`kite/robot/alicia.py`、数据标签、proprio 和文档已统一改为 interleaved。

## 训练数据

- 120 个脚本示教 episode（102 train / 18 val），20 Hz，231 步/集；
- 动作：绝对 EEF `[xyz(m), rot6d(6), gripper(1)]`，夹爪 `1=闭合`；
- 数据：`outputs/alicia_episodes_v2`，特征：`outputs/alicia_features_v2` / `alicia_features_full_v2`；
- 示教使用临时“方块吸附”辅助（`grasp_assist=true`），真实物理抓取仍未解决。

## 当前结果

- 2 epoch interleaved 微调：val `position 0.555 / rot 0.00058 / gripper 0.128`；
- 纯策略（无固定朝向/偏移/吸附）能接近方块并闭合夹爪，随后按示教移动到落点；
- **MuJoCo 物理抓取仍失败**：方块未被夹住，原因是 50 mm 指爪接触/几何，而不是策略不闭合；
- 带 `--grasp-assist` 时可完成搬运，但这只验证链路，不是物理抓取。

## 使用

```powershell
python scripts/alicia_xvla_realtime.py `
  --checkpoint weights/X-VLA-WidowX-alicia/softprompt_domain19.pt `
  --domain-id 19 --headless --cycles 12 --exec-steps 10
```

训练/评估约定见 [`docs/alicia-d-native.md`](../../docs/alicia-d-native.md) §7。
