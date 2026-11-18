import time

import numpy as np
import tcim_lite as tcim

ENC = "/home/kylin/kite-work/hmm/xvla_encoder.hmm"
DEN = "/home/kylin/kite-work/hmm/xvla_denoiser.hmm"

encoder = tcim.runtime.Module.load(ENC)
denoiser = tcim.runtime.Module.load(DEN)
print("loaded encoder/denoiser")

rng = np.random.default_rng(0)
input_ids = np.zeros((1, 50), dtype=np.int32)
pixel_values = rng.standard_normal((1, 2, 3, 224, 224)).astype(np.float16)
encoder.set_input("input_ids", input_ids)
encoder.set_input("pixel_values", pixel_values)
start = time.perf_counter()
encoder.run()
encoder.sync()
enc_ms = (time.perf_counter() - start) * 1000
enc = {}
for i in range(encoder.get_num_outputs()):
    name = encoder.get_output_name(i)
    enc[name] = encoder.get_output(name).astype(np.float32).numpy()
print("encoder", {k: v.shape for k, v in enc.items()}, f"{enc_ms:.1f} ms")

x1 = rng.standard_normal((1, 30, 20)).astype(np.float16)
action = np.zeros((1, 30, 20), dtype=np.float16)
proprio = np.zeros((1, 20), dtype=np.float16)
start = time.perf_counter()
for step in range(10):
    t = np.array([(step + 1) / 10], dtype=np.float16)
    x_t = x1 * t.reshape(1, 1, 1) + action * (1 - t).reshape(1, 1, 1)
    x_t[..., 9] = 0
    x_t[..., 19] = 0
    denoiser.set_input("vlm_features", enc["vlm_features"].astype(np.float16))
    denoiser.set_input("aux_visual_inputs", enc["aux_visual_inputs"].astype(np.float16))
    denoiser.set_input("action_with_noise", x_t)
    denoiser.set_input("proprio", proprio)
    denoiser.set_input("t", t)
    denoiser.run()
    denoiser.sync()
    action = denoiser.get_output("action").astype(np.float32).numpy()
    x1 = action.astype(np.float16)
den_ms = (time.perf_counter() - start) * 1000
out = action
print("denoiser", out.shape, f"{den_ms:.1f} ms", "range", float(out.min()), float(out.max()))
print("gripper", out[0, 0, [9, 19]])
