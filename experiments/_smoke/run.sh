#!/bin/bash
# Smoke test: confirms the job wrapper, GPU and results path work.
python - <<'PY'
import json, os, time, torch
t = time.time()
x = torch.randn(8192, 8192, device="cuda")
y = (x @ x).sum().item()
out = {
    "gpu": torch.cuda.get_device_name(0),
    "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    "matmul_s": round(time.time() - t, 3),
    "torch": torch.__version__,
}
json.dump(out, open(os.environ["RUN_DIR"] + "/out/metrics.json", "w"))
print(out)
PY
