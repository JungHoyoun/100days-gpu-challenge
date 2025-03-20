### online softmax algorithm

The onlinesoftmax.cu now is the kernel from Nvidia

```
Algorithm: Online Softmax
1: m0 ← -∞
2: d0 ← 0
3: for j ← 1 to V do
4:     mj ← max(mj-1, xj)
5:     dj ← dj-1 * e^(mj-1 - mj) + e^(xj - mj)
6: end for
7: for i ← 1 to V do
8:     yi ← e^(xi - mV) / dV
9: end for
```