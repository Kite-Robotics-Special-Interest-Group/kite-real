# Alicia-D 真机 / 仿真实验

从 `Kite` 仓迁移出的真机研究子项目：Alicia-D v5.6 SDK 适配、MuJoCo 仿真、
示教采集、domain 19 soft prompt 训练和闭环回归。

## 目录

```text
alicia/
  kite/robot/                 # Alicia SDK/单位/动作适配（运行时也可复用）
  kite/sim/                   # MuJoCo 场景、相机、绝对 EEF 控制器
  configs/robot/              # 机器人/相机/场景/训练配置
  scripts/                    # 采集、特征、训练、实时闭环、约定检查
  docs/alicia-d-native.md     # 完整说明
  tests/                      # 接口与仿真测试
  weights/X-VLA-WidowX-alicia/README.md
```

## 环境与资产

```bash
cd alicia
pip install -r requirements.txt
python scripts/fetch_alicia_assets.py --proxy http://127.0.0.1:7897
```

模型/权重、`assets/`、`third_party/`、`outputs/` 不进 Git。

## 运行

```bash
# 接口/单位/仿真测试
python -m pytest -q

# 纯策略闭环（不吸附）
python scripts/alicia_xvla_realtime.py \
  --checkpoint weights/X-VLA-WidowX-alicia/softprompt_domain19.pt \
  --domain-id 19 --headless --cycles 12 --exec-steps 10

# 演示：固定朝向 + 标定偏移 + 示教吸附
python scripts/alicia_xvla_realtime.py \
  --checkpoint weights/X-VLA-WidowX-alicia/softprompt_domain19.pt \
  --domain-id 19 --grasp-assist --fixed-orientation \
  --position-offset 0 -0.02 -0.015 --exec-steps 30 --duration 120
```

## 手柄遥操作采集

```bash
# 打开可视化窗口；首次输入自动开始记录
python scripts/alicia_teleop_collect.py

# 指定预分配指令（configs/teleop_tasks.yaml 的索引）
python scripts/alicia_teleop_collect.py --task-index 1

# 保存后自动切到下一条指令
python scripts/alicia_teleop_collect.py --cycle-tasks

# 方块抬升后自动判定完成（默认手动 A 完成）
python scripts/alicia_teleop_collect.py --auto-success

# 只检查手柄（XInput + winmm fallback）
python scripts/alicia_teleop_collect.py --self-test
```

- 指令标签：`configs/teleop_tasks.yaml`（如“拿起红色方块 / pick up the red block”）；D-pad 左右或键盘 `N/P` 切换，`--task-index` 指定
- 左摇杆：base X/Y 平移；右摇杆：Z / yaw
- RT/LT：夹爪闭合 / 张开；A/Start：任务完成，弹窗询问是否保存；B：取消
- LB：取消并重置；RB：切换 free/front 相机；D-pad 左右切指令
- 鼠标：左键旋转、右键平移、滚轮缩放；V 也可切换相机
- 手柄状态默认在**独立窗口**，不遮挡 3D 画面；`--gamepad-panel hud/off` 可切回叠加/关闭
- 接触摩擦已调高；默认开启演示吸附，`--no-grasp-assist` 可关闭
- 数据：`outputs/alicia_teleop/<timestamp>/`（action/proprio/obs/meta.json）
- episode 级划分：`python scripts/make_teleop_split.py --root outputs/alicia_teleop`，写入 `splits.json` 并更新 `meta.split`；长 episode 默认不进 val
- 无手柄时可用键盘：W/S x、A/D y、Q/E z、Z/X yaw、O/C 夹爪、T 完成

## 当前边界

- rot6d 已对齐官方行交错 `[b1x,b2x,b1y,b2y,b1z,b2z]`；
- 纯策略能接近、闭合、移向落点，但 MuJoCo 50 mm 指爪仍夹不住方块；
- 物理抓取修复前，`--grasp-assist` 只用于链路演示，不作为成功率。

论文 SIMPLER / LeRobot 复现代码不在此仓，也不在 Kite 仓。
