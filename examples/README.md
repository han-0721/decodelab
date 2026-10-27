# 本机 CPU 示例

`cpu-smoke.json` 和 `cpu-smoke.csv` 来自一次真实执行，用于展示报告格式。
硬件、软件、种子、模型参数与原始 trial 都在 JSON 中。
此次使用 CPU，PyTorch 为 CPU 构建；未进行 4090 性能测试。

复现命令：

```bash
python -m decodelab benchmark --prompt-len 64 --new-tokens 32 --warmup 2 --repeats 5 --output results/cpu-smoke
```

CPU 调度、后台负载、PyTorch 版本和硬件都会影响绝对数值。
这些结果用于展示实验方法，无法推导其他硬件或成熟推理引擎的性能。
后续功能改变后应重新运行并保存新的报告；保留原始 trial。
