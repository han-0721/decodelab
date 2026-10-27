# DecodeLab

**From `forward` shapes to exact KV caching — a small, readable inference lab.**

从一次 `forward` 看懂 token 如何生成，再用实验比较三种 KV Cache 路径。
项目使用原生 PyTorch 实现一个小型 decoder，并把模型、缓存、生成循环和计时分开。

**当前为学习与数值验证版本 v0.1：模型权重是固定随机种子初始化的，未经训练。**
输出 token ID 用于检查计算过程，不代表自然语言生成能力；运行时无需下载模型。
仓库中的实验不代表 vLLM、生产服务或 RTX 4090 的性能。

## 从这里开始

需要 Python 3.10+。先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
选择适合机器的 CPU/CUDA 版本，再安装本项目。

```bash
git clone https://github.com/han-0721/decodelab.git
cd decodelab
python -m venv .venv
```

激活环境：Linux/macOS 使用 `source .venv/bin/activate`；
Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`。
也可直接调用虚拟环境中的 Python，避免调整系统执行策略。

CPU 环境可执行：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m decodelab trace --prompt 1,2,3,4
python -m decodelab demo --prompt 1,2,3,4 --new-tokens 8 --cache dynamic
python -m decodelab check
python -m pytest -q
```

`trace` 会分别展示 prefill 与一次 cached decode 的 shape。
`check` 比较全序列/逐 token 的 logits，并检查三种模式的贪心生成是否一致。
所有命令均默认在 CPU 执行；安装 CUDA 版 PyTorch 后，可显式增加 `--device cuda`。

## 一次 forward 到底做了什么

以 `B=1, T=4, D=128, H=4, V=256` 为例：

```text
token IDs                  [1, 4]
    ↓ token + position embedding
hidden states              [1, 4, 128]
    ↓ split Q / K / V into heads
Q, K, V                    [1, 4, 4, 32]   = [B, H, T, D/H]
    ↓ causal attention + MLP, repeated for each block
hidden states              [1, 4, 128]
    ↓ final norm + output projection
logits                     [1, 4, 256]
    ↓ take last position, then argmax
next token                 [1, 1]
```

`forward` 负责计算分数；`generate` 在模型外循环选择 token。
prefill 处理整个 prompt，建立各层 K/V；随后每次 decode 输入一个新 token，
让它的 Q 关注历史和当前 K/V。最后一个刚选出的 token 尚未再次送入模型。

## 三条可以直接比较的路径

| 模式 | 每次送进模型的输入 | K/V 存储方式 | 想研究的问题 |
|---|---|---|---|
| `none` | prompt + 已生成 token | 每轮重算 | 全序列重复计算有多少代价？ |
| `dynamic` | 首轮完整 prompt，后续一个 token | 每层通过 `torch.cat` 扩展 | 缓存复用带来什么收益，扩容有什么代价？ |
| `static` | 与 dynamic 相同 | 首次预分配容量，之后按位置写入 | 固定容量与动态分配有什么取舍？ |

三条路径使用相同模型权重和贪心策略。缓存保存精确的浮点 K/V，
没有淘汰历史、量化或近似注意力。浮点运算路径不同仍可能出现微小数值误差，
因此测试同时比较 logits 容差与固定样例的生成结果。

## 自己跑一个实验

```bash
python -m decodelab benchmark --prompt-len 64 --new-tokens 32 --warmup 2 --repeats 5 --output results/cpu-p64
```

之后可以在 4090 上使用同一套命令：

```bash
python -m decodelab check --device cuda
python -m decodelab benchmark --device cuda --dim 512 --layers 4 --heads 8 --context 1024 --prompt-len 256 --new-tokens 64 --warmup 3 --repeats 10 --output results/4090-p256
```

命令生成完整 JSON（环境、参数、原始 trial、ITL 样本、摘要）和摘要 CSV。
从 prompt 长度、输出长度、batch size 三个维度逐个改变变量，记录观察和解释。
小模型中 Python 调度、输入检查与同步开销可能占主导；静态缓存也不保证总是更快。

| 指标 | 本项目的含义 |
|---|---|
| TTFT | 生成调用开始到第一个新 token 选出且设备计算完成，不含分词/网络 |
| ITL | 相邻两次新 token 选出完成之间的时间；batch>1 时是整个 batch 的步间隔 |
| Output tok/s | `batch_size × new_tokens / 整次生成耗时`，包含 prefill |
| Cache used bytes | 已写入有效 K/V 的逻辑字节数 |
| Cache allocated bytes | 缓存 tensor 的容量字节数，排除临时副本及 allocator 额外开销 |
| CUDA peak allocated | PyTorch 活跃 tensor 峰值，包含模型和输入；CPU 为 null |

完整计时口径见 [实验方法](docs/benchmarking.md)。

## 阅读顺序

1. [第一次读源码](docs/first-steps.md)：边打印 shape 边读，一次只跟一条请求。
2. [缓存如何工作](docs/kv-cache.md)：位置、mask、预分配和最后一个 token。
3. [实验方法](docs/benchmarking.md)：设计可复现对照，区分计时口径。
4. [后续路线](docs/roadmap.md)：从学习项目逐步长成独立的推理优化项目。

```text
src/decodelab/
  model.py        # 小型 decoder：embedding、attention、MLP、logits
  cache.py        # dynamic / static KV 存储
  generation.py   # 三种模式共享的生成循环与计时
  benchmark.py    # 多次测量、环境信息、JSON/CSV 导出
  cli.py          # demo / trace / check / benchmark
tests/            # 数值、因果性、缓存、生成与 CLI 测试
docs/             # 中文学习指南与实验说明
examples/         # 实测 CPU 示例与复现口径
```

## 当前边界与后续贡献

v0.1 已实现单设备、同长度 batch、完整因果注意力与 learned position embedding。
后续可以逐步加入真实预训练权重适配、变长请求、continuous batching、分页 KV 管理，
再研究自己的调度或缓存策略。每增加一个机制，先建立基线和正确性检查，
再记录实验结果；已完成项与计划项在 [路线图](docs/roadmap.md) 中分开维护。

## 参考与许可

这是本仓库独立编写的紧凑实现，理论与学习资料包括：

- [PyTorch：构建神经网络](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Hugging Face：How caching works](https://huggingface.co/docs/transformers/cache_explanation)
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：后续阅读更完整推理引擎的参考
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：后续分页缓存阶段的参考

MIT License。未包含第三方模型权重。
