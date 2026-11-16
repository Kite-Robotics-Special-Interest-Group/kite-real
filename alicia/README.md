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

## 当前边界

- rot6d 已对齐官方行交错 `[b1x,b2x,b1y,b2y,b1z,b2z]`；
- 纯策略能接近、闭合、移向落点，但 MuJoCo 50 mm 指爪仍夹不住方块；
- 物理抓取修复前，`--grasp-assist` 只用于链路演示，不作为成功率。

论文 SIMPLER / LeRobot 复现代码不在此仓，也不在 Kite 仓。
