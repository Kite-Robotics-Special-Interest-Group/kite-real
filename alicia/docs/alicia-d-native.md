# Alicia-D v5.6 原生 X-VLA 链路（实机 + 仿真 + 监督）

> 目标：回到 **X-VLA 原生 PyTorch 方案**，把 Alicia-D v5.6（50 mm 夹爪）的
> 实机 SDK、URDF/MJCF、MuJoCo 仿真和人工监督窗口搭好，并把数据 / 接口 / 单位约定固定下来。
>
> 本文只描述 **Alicia-D 单臂 + 10 维 EEF 动作**。
> 论文 SIMPLER / LeRobot 复现代码不在本仓，也不在 Kite 仓。

## 1. 当前状态

| 项 | 状态 | 位置 |
| --- | --- | --- |
| Alicia-D-SDK（v6.1.0rc5） | 已克隆 | `third_party/Alicia-D-SDK`（commit `7c49e07`） |
| URDF / MJCF / 9 个 STL | 已下载并校验 | `assets/synriard/` |
| X-VLA-WidowX 权重 | 已就绪 | `weights/X-VLA-WidowX/model.safetensors`（3.52 GB，903 张量） |
| MuJoCo 场景 + 双相机 + 绝对 EEF 控制 | 已跑通 | `kite/sim/alicia.py` |
| 人工监督窗口 | 已跑通 | `scripts/alicia_sim_viewer.py` |
| Alicia soft prompt | 已训练 domain 19 | `weights/X-VLA-WidowX-alicia/softprompt_domain19.pt` |
| 数据 / 接口 / 单位检查 | 见 `outputs/alicia_checks.json` | `scripts/check_alicia_conventions.py` |

两个 WARN 是**本体级事实**，不是脚本错误：

1. `domain_id=0` 是 Bridge/WidowX 域；Alicia-D **没有**在该 checkpoint 里训练过，
   零样本动作不可信；
2. processor 期望 3 视角；仿真/实机目前提供 `front + wrist`，第 3 视角补 0。

两个 SKIP：

- 本机没有 `data/`，数据检查跳过；
- 本机 Python 3.14 未安装 vendor 依赖，`alicia_d_sdk` 导入检查跳过；真机按 SDK 支持的环境安装。

## 2. 资产与 SDK

```powershell
# 1) 下载 URDF/MJCF/STL（支持断点续传），并克隆 SDK
python scripts/fetch_alicia_assets.py --proxy http://127.0.0.1:7897

# 2) 真机侧安装 SDK（建议 Python 3.9–3.13）
python -m pip install -e third_party/Alicia-D-SDK

# 3) 只读自检（不指定 --home 不会动）
python scripts/alicia_real_check.py --port COM3
```

`fetch_alicia_assets.py` 会在 `assets/synriard/manifest.json` 记录每个文件的
大小与 SHA256，以及 SDK commit。资产和 `third_party/` 均**不进 Git**。

SDK 关键事实：

| 项 | 约定 |
| --- | --- |
| 关节 | 6 旋转关节 `Joint1..Joint6`，内部弧度；`set_robot_state(joint_format="deg")` 可传角度 |
| 夹爪 | `gripper_value` 整数 `0..1000`；**1000=张开，0=闭合** |
| 位姿 | `[x, y, z, qx, qy, qz, qw]`，米 + 四元数 xyzw |
| 坐标系 | `base_link -> tool0`；`tool0` 在 `link6` 本地 `(0, 0, -0.13)` |
| 速度 | `speed_deg_s` 单位是 deg/s；关节内部仍是弧度 |
| 状态 | `get_robot_state("joint_gripper")` 返回 `angles`（rad）和 `gripper`（0–1000） |

## 3. 仿真环境

`kite/sim/alicia.py` 直接加载上游 MJCF，并在编译前补充：

- 8 个位置执行器（6 关节 + 2 指），`armature=0.05`、`damping=0.5`，避免低惯量数值发散；
- 固定 `front` 相机（第三人称）和 `link6` 上的 `wrist` 相机（侧装，看向夹爪/桌面）；
- 桌面、方块、黄色 `eef_target` 目标球；
- `tool0_site` 作为 TCP，与 SDK 的 `end_link=tool0` 对齐。

控制链路：

```
X-VLA 绝对动作 [xyz(3), rot6d(6), grip(1)]
  → 限幅（工作空间 / 单步位移 / 单步旋转 / gripper）
  → MuJoCo 阻尼最小二乘 IK（6 关节，位置 + 姿态）
  → 位置执行器 → mj_step
```

仿真按配置 `20 Hz` 控制、`0.002 s` 物理步长、每控制步 25 次子步。

### 监督窗口

```powershell
# 图形窗口：默认自动演示，可随时接管
python scripts/alicia_sim_viewer.py

# 手动模式（不自动动）
python scripts/alicia_sim_viewer.py --no-auto

# 无窗口冒烟 + 保存 front/wrist 拼图（--no-auto 先停在初始位）
python scripts/alicia_sim_viewer.py --headless --no-auto --frames 100 `
  --snapshot outputs/alicia_sim.png
```

| 按键 | 作用 |
| --- | --- |
| `space` | 暂停 / 继续 |
| `b` | 刹车：目标锁到当前 TCP |
| `r` | 重置场景 |
| `h` | 目标回到初始位 |
| `w/s` | 沿 base X 后退 / 前进 |
| `a/d` | 沿 base Y 左 / 右 |
| `q/e` | 沿 base Z 上 / 下 |
| `z/x` | 目标偏航 - / + |
| `o` / `c` | 张开 / 闭合夹爪 |
| `g` | 切换夹爪 |
| `t` | 自动演示 / 手动目标 |
| `p` | 打印当前状态 JSON |

窗口左下角叠加腕部相机画面；`--no-wrist-inset` 可关闭。

### 安全限幅

`kite/robot/alicia.py` 的 `clamp_action10` 在动作进入仿真或实机前执行：

- 工作空间：`x∈[0.05, 0.65]`、`y∈[-0.35, 0.35]`、`z∈[0.02, 0.75]` 米；
- 单控制步位移 ≤ 30 mm、旋转 ≤ 0.20 rad；
- gripper 裁到 `[0, 1]`。

干预项会返回 `interventions`，监督窗口和实机客户端都可用它记录“人类/安全层是否改过模型动作”。

## 4. X-VLA 接口约定

| 项 | 值 |
| --- | --- |
| 动作张量 | `[B, 30, 20]`，Alicia 只用前 10 维 |
| 动作布局 | `[x, y, z, rot6d(6), gripper(1)]` |
| 位置 | 米，`base_link` 系，**绝对 EEF** |
| 朝向 | 旋转矩阵前两列，**行交错** `[b1x,b2x,b1y,b2y,b1z,b2z]`；转 SDK 四元数 xyzw |
| 夹爪 | 模型 `1=闭合`、`0=张开`；SDK `0=闭合`、`1000=张开` |
| proprio | 20 维：`[pos(3), rot6d(6), 0] + 10×0`（夹爪通道被模型预处理置零） |
| 图像 | RGB、224×224、ImageNet mean/std |
| 视角 | `front` 主视角、`wrist` 辅助；第 3 视角补 0 |
| domain | 当前 checkpoint 固定 Bridge/WidowX `domain_id=0` |

转换函数：

```python
from kite.robot.alicia import action10_to_sdk_pose, sdk_pose_to_action10

pose7, gripper_ticks = action10_to_sdk_pose(action10)
# pose7 = [x, y, z, qx, qy, qz, qw]
```

### 4.1 必须承认的本体差异

X-VLA-WidowX 学的是 **Bridge/WidowX** 的视觉、TCP、相机和动作分布。Alicia-D：

- 是新本体，`domain_id=0` 的 soft prompt / 动作头不是为它训练的；
- `tool0` 与 WidowX 训练时的 EEF 参考点不同；
- 相机安装、视野、桌面高度都需要重新标定。

因此当前链路能保证的是：**SDK / 仿真 / 单位 / 动作布局 / 安全限幅正确**；
**不能保证零样本任务成功率**。要获得可用策略，需要在 Alicia-D 数据上微调
（新增 domain 或复用/扩展 `domain_id`），并保持训练与部署的相机、TCP、夹爪极性一致。

### 4.2 实机运行顺序

```python
from kite.robot.alicia import AliciaSdkClient

client = AliciaSdkClient(port="COM3", auto_connect=True)
client.connect()
obs = client.read()          # joints / TCP / 20D proprio / 10D action
result = client.send_action10(action10, speed_deg_s=10, previous=obs)
print(result.interventions)
client.hold()                # 监督停车：保持当前关节
client.close()
```

`send_action10` 默认会执行动作；调试时先跑 `alicia_real_check.py`，确认
`joint_rad`、`tcp_position_m`、`gripper_ticks` 与实机一致，再启用运动。

## 5. 数据约定

本地数据目录（`data/`）不提交 Git。`check_alicia_conventions.py --data-root data/...`
会检查：

- 每条 episode 的 JSON meta 必须包含：
  `episode_id`、`fps`、`instruction`、`camera_intrinsics`、`camera_extrinsics`、
  `depth_unit`、`renderer`、`action_layout`、`units`、`split`；
- 动作 `.npy` 必须是 `[T, 10]`；
- `data/splits.json` 必须有互斥的 `train` / `val`；
- 单位固定为：长度米、角度弧度、深度米、夹爪 `0..1` 且 `1=闭合`。

相机内参/外参不进入模型，但必须与训练数据一起归档；训练和部署使用同一安装位置。

## 6. 一键检查

```powershell
python scripts/check_alicia_conventions.py --json outputs/alicia_checks.json
# WARN 也想失败时：
python scripts/check_alicia_conventions.py --strict
```

当前本机结果（2026-11-13）：

```
PASS=34  WARN=2  FAIL=0  SKIP=2
```

WARN：`cross:domain`、`cross:views`。详见 `outputs/alicia_checks.json`。

## 7. Alicia soft prompt 训练

原版 `domain_id=0` 在 Alicia-D 上跑闭环时，位置能大致朝方块移动，但夹爪不闭合、任务失败。
因此为 Alicia-D 新建 **domain 19**：冻结 Florence2 和 24 层 Transformer，仅训练
domain 19 的 soft prompt、action encoder 和 action decoder 行。

### 7.1 采集

```powershell
python scripts/collect_alicia_episodes.py --episodes 120 --out outputs/alicia_episodes
```

- 120 个成功 episode，每个 231 步（20 Hz），`action.npy` 为绝对 EEF `[T,10]`；
- `obs/` 每 2 步保存 front/wrist JPEG（训练时按 20 Hz 动作对齐）；
- 指令 `pick up the red block`；方块/立柱位置随机；
- 脚本专家在闭合后临时把方块挂到 TCP，避免 vendor 50 mm 指爪在 MuJoCo 里滑落；
  `meta.json` 中 `grasp_assist=true`，这是**示教辅助**，不是物理抓取验证。

### 7.2 特征缓存

```powershell
python scripts/precompute_alicia_features.py --train-limit 8000 --val-limit 1500
```

缓存冻结 VLM 的 `vlm_features` 与 `aux_visual`（fp16，各 100×1024），训练时只跑小头。

### 7.3 训练

```powershell
python scripts/train_alicia_softprompt.py `
  --features outputs/alicia_features_v2 `
  --domain-id 19 --epochs 3 --batch 4 --lr 3e-4 --gripper-weight 5 `
  --out weights/X-VLA-WidowX-alicia/softprompt_domain19.pt
```

- 初始化：domain 19 从 domain 0 复制（soft prompt + 三个 action 头）；
- 可训练：domain 19 的 32×1024 soft prompt + action encoder/decoder 行；
- 其余 domain 和全部共享参数冻结；`weight_decay=0` 避免未训练行漂移；
- 夹爪 BCE 权重 5，缓解“位置学会了但夹爪不闭合”。

### 7.4 闭环查看

```powershell
python scripts/alicia_xvla_realtime.py `
  --checkpoint weights/X-VLA-WidowX-alicia/softprompt_domain19.pt `
  --domain-id 19 --grasp-assist --fixed-orientation `
  --position-offset 0 -0.02 -0.015 --exec-steps 30 --duration 120
```

`--grasp-assist` 只在演示时把方块吸附到 TCP；真实物理抓取仍需继续调指爪几何/摩擦。
不加 `--grasp-assist` 时，脚本执行纯策略动作，可用于观察位置/夹爪预测。

当前训练/评估记录：

| 项 | 值 |
| --- | --- |
| domain | 19（初始化自 0） |
| 训练样本 | 8000（train 102 episodes） |
| 验证样本 | 1500（val 18 episodes） |
| 有效可训练行 | 约 128k 参数（优化器张量 3.84M） |
| 当前 checkpoint | 2 epoch（interleaved，base domain 0 初始化） |
| 最近 val loss | position 0.555 / rot 0.00058 / gripper 0.128（`outputs/alicia_softprompt_log_v2.json`） |
| checkpoint | `weights/X-VLA-WidowX-alicia/softprompt_domain19.pt` |

### 7.5 闭环观察

- **纯策略**（无固定朝向、无偏移、无吸附）：TCP 能从 home 移到方块附近
  （约 13–20 mm），但夹爪长期保持张开，任务不完成；这是当前 soft prompt 的边界。
- **加部署标定**（`--fixed-orientation --position-offset 0 -0.02 -0.015`，**仍无吸附**）：
  第 3 个动作块夹爪已闭合，但 TCP `[0.432, 0.056, 0.301]` 与方块
  `[0.421, 0.103, 0.300]` 在 y 方向差 4.7 cm，夹空；方块没有随夹爪抬升。
  这说明当前模型能学会“到高度 → 闭合”的阶段，但平面定位仍有偏差。
- **纯物理抓取结论**：不加 `--grasp-assist` 时**不能正常夹取**；
  仅加吸附辅助才完成搬运。
- **加示教吸附**（`--grasp-assist`）：全量微调后，策略在第 7 个 30 步动作块内完成
  “接近 → 闭合 → 抬升 → 移到 (0.34, -0.14) → 张开释放”，最终方块落到桌面 `z≈0.24`。
  这验证了 soft prompt、动作块和闭环链路，但**不是物理抓取验证**。
- 全量数据把释放提前到第 7 块（此前约第 11 块），但纯策略的夹爪仍未可靠闭合；
  说明主要瓶颈是示教抓取分布与部署状态偏移，而不是模型容量。

### 7.6 2026-11-15 rot6d 修正（重要）

官方 X-VLA 的 rot6d 是**行交错**：

```text
[b1x, b2x, b1y, b2y, b1z, b2z]
```

不是 `[b1x,b1y,b1z,b2x,b2y,b2z]`。之前的 Alicia 数据、proprio、adapter 和
`docs/xvla-model-io.md` 都用了后者，导致模型朝向和平面定位偏差。

已做：

- `kite/robot/alicia.py` 改为 interleaved；
- 旧 episode/feature 标签用 `scripts/convert_alicia_rot6d.py` 重排；
- 新数据：`outputs/alicia_episodes_v2`、`outputs/alicia_features_v2`、
  `outputs/alicia_features_full_v2`；
- 从 base domain 0 重新训练 domain 19（2 epoch，interleaved），
  val `position 0.555 / rot 0.00058 / gripper 0.128`；
- 旧 column-major checkpoint 保留为
  `weights/X-VLA-WidowX-alicia/softprompt_domain19_columns.pt`。

闭环变化：纯策略现在**会闭合夹爪**，并按“接近 → 闭合 → 移向落点”执行；
但 MuJoCo 里方块仍未被夹住（50 mm 指爪接触/几何问题），
所以**物理抓取还不能算通过**。已尝试加 finger collision pads 并做 108 组
参数扫描，均未成功；`grasp_pads` 默认关闭，保持与训练图像一致。

### 7.7 停机前状态

- 最后 checkpoint：`softprompt_domain19.pt`（2 epoch interleaved）；
- 全量数据（11,832/2,088）微调跑到第一个 epoch 前被停机，未保存新权重；
- 没有后台训练/仿真进程；GPU 已释放；
- 继续训练命令：

```powershell
python scripts/train_alicia_softprompt.py `
  --features outputs/alicia_features_full_v2 `
  --epochs 2 --batch 4 --lr 2e-4 --limit 11832 --val-limit 2088 `
  --gripper-weight 10 --resume weights/X-VLA-WidowX-alicia/softprompt_domain19.pt `
  --out weights/X-VLA-WidowX-alicia/softprompt_domain19.pt `
  --log outputs/alicia_softprompt_log_full_v2.json
```

## 8. 下一步

1. 把 `--grasp-assist` 从演示路径移除：修正 50 mm 指爪在 MuJoCo 的接触/摩擦，或换更小方块；
2. 扩充数据到真实相机/真实桌面，保持 episode 级 train/val；
3. 训练 3–5 epoch，比较开环动作误差与闭环成功率；
4. 实机先用 `--action10` 校验单步动作，再放开连续 `/act` 推理；
5. 需要端侧时，再把同一接口导出到 ONNX / `.hmm`（见 M50 文档）。
