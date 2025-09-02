import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=1000000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--input", type=str, default="input.txt")
parser.add_argument("--ref", type=str, default="ref.txt")
args = parser.parse_args()

torch.manual_seed(args.seed)

# 1D 텐서 생성 (값 분포는 자유롭게 조절 가능)
x = torch.randn(args.N, dtype=torch.float32) * 0.5 + 1.0  # 예시 분포

# softmax: 1D 벡터에 대한 안정화된 softmax
# softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
x_max = x.max()
ex = torch.exp(x - x_max)
y = ex / ex.sum()

# 텍스트로 저장 (한 줄에 하나씩)
np.savetxt(args.input, x.numpy(), fmt="%.8f")
np.savetxt(args.ref, y.numpy(), fmt="%.8f")

print(f"Wrote {args.input} (N={args.N}) and {args.ref}")
