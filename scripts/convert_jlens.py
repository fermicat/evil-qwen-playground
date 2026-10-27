"""Convert a Jacobian lens file (Anthropic's jlens format, torch) to .npz for MLX. Run once; torch is not a project
dependency:

    uv run --with torch python -I scripts/convert_jlens.py \
        models/jlens-download/Qwen3.5-4B_jacobian_lens_n1000.pt models/jlens/qwen35-4b-n1000.npz
"""

import sys

import numpy as np
import torch

src, dst = sys.argv[1:3]
ck = torch.load(src, map_location="cpu", weights_only=True)  # tensors only, no pickled code
J = {f"L{k}": v.to(torch.float16).numpy() for k, v in ck["J"].items()}
np.savez(dst, n_prompts=ck["n_prompts"], **J)
print(f"{dst}: layers {sorted(ck['J'])}, d_model {ck['d_model']}, n_prompts {ck['n_prompts']}")
