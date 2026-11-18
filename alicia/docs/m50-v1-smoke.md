# M50（XH2 V1）板端冒烟记录

日期：2026-11-18  
板子：`kylin-pc` / Kylin V10 aarch64 / 100.76.124.84  
运行时：`tcim_lite v1.5.0`（驱动/固件 V1.4.0）  
编译器：Dadao 1.5.0，`target=xh2`、`march=v1`、`ncore=2`

## 产物

- `xvla_encoder.hmm` 671,523,472 B，SHA256 `756c7305F6A6D997AEE65AED1FBB2270DB7926E7CE029B8D32582F639ECD9B80`
- `xvla_denoiser.hmm` 327,108,448 B，SHA256 `186525A078A26D8631EDF6FE1642E77FEBF7CFE75163C416B93A918FB874A44E`
- 均为 domain 19、2 视角、`input_ids` 长度 50。

## 板端结果

```
encoder: vlm_features [1,100,1024], aux_visual_inputs [1,50,1024], 30.7 ms
denoiser: 10 步, action [1,30,20], 132.4 ms
```

完整 pipeline 约 163 ms（不含图像预处理/后处理）。
随机输入仅用于形状/驱动验证，不代表策略精度。

## 注意事项

- 旧 V2 产物在新板直接报 `compiled using incompatible V2`，必须用 `march=v1` 重编。
- 传输大文件不要走 12 路并发；DERP 会掉 SSH。可用云端中继：
  `221.5.60.34:/root/hmm-relay` → 板端拉取，实测约 3 MB/s。
- 板端运行：

```bash
source ~/下载/houmo_tcim_runtime_xh2_linux_aarch64-1.5.0/env.sh
cd ~/kite-work
python3 scripts/board_hmm_smoke.py
```
