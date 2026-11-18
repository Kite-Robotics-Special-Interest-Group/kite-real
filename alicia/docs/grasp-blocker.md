# Alicia-D 50 mm 指爪抓取排查记录

日期：2026-11-17  
环境：云 RTX 5090 / MuJoCo 3.17 / 原版 Alicia MJCF

## 结论

当前 vendor MJCF 的 50 mm 平行指爪在 MuJoCo 中无法稳定夹住方块。
策略已经能学会接近、闭合和移动到落点；失败点在**物理接触**，不是 soft prompt 或训练数据。

## 已尝试

- 原始 finger mesh collision：闭合后方块被推开，抬升时滑落。
- 增加 finger collision pads（box），改变 pad 尺寸、位置、摩擦、condim、执行器力。
- 对称/世界对齐 pad、不同 cube 尺寸（2.5/3/4 cm）、不同 pedestal 高度。
- 132+ 组参数组合；成功数为 0。
- 典型现象：
  - pad 与 pedestal 先接触，方块被推离平台；
  - 左右 pad 的接触点不对称，闭合时方块向上/侧向挤出；
  - 闭合后 `qpos` 已到限位，但方块 `z` 不上升。
- 关掉 `grasp-assist` 时，策略可以闭合夹爪，但方块仍留在台面。

## 云训练状态

- 13,920 帧特征缓存完成；
- 全量 11,832/2,088，3 epoch + 3 epoch（gripper weight 30）；
- val `position 0.501 / rot 0.00021 / gripper 0.102`；
- checkpoint 已同步：
  `D:\JLU_tools\kite-real\alicia\weights\X-VLA-WidowX-alicia\softprompt_domain19.pt`

## 阻塞原因

MuJoCo 里 vendor 50 mm 指爪的接触几何/接触点与方块不匹配。
继续训练只能改善动作预测，不能改变物理指爪是否能夹住；不能用 `grasp-assist` 冒充成功率。

## 解除阻塞所需

1. 用真实的 50 mm 指爪 CAD/碰撞体替换当前 finger collision，或使用经验证的平行夹爪模型；
2. 在原生 MuJoCo/真机上验证 oracle 能稳定夹起 3 种方块；
3. 重新采集无辅助示教并重训，再跑 held-out 成功率。

在上述物理模型验证前，不报告可用权重。
