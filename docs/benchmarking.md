# 实验口径与复现方法

## 先确定实验问题

第一组问题：复用 KV 后，整段重算的代价如何随输入/输出长度变化？
第二组问题：精确缓存中，dynamic 扩容与 static 预分配的取舍是什么？

建议一次改变一个变量：

| 变量 | 起步取值 |
|---|---|
| prompt length | 16、64、256 |
| new tokens | 16、32、64 |
| batch size | 1、2、4 |

各组实验固定模型维度、层数、head 数、精度、设备、线程、权重和 prompt 种子。
确保 `context >= prompt_length + new_tokens`。
模型随机权重只用于控制计算量和验证数值；不会反映真实语义、EOS 分布或模型质量。

## 计时覆盖范围

模型创建、迁移到设备、prompt 生成在计时之外。
每次调用在开始前同步 CUDA；计时包含 prompt 克隆、缓存初始化及首次分配、
所有模型调用、输入验证、greedy 选择、输出追加和逐步同步。
每选出一批 token 后同步设备再记录时间，确保计时反映完成的设备工作。

TTFT 是调用开始到首 token 选出完成的时间，包含 prefill，排除网络和分词。
ITL 来自相邻完成时间之差；一个 trial 生成 N 个 token，有 N-1 个 ITL。
batch>1 时，ITL 是 batch 步间隔，吞吐则按所有序列的输出 token 总数计算。
本项目测量的是推理循环微基准，没有请求到达队列，不报告 Serving Goodput 或 SLO。

缓存模式分别 warmup，每次 measured trial 轮换模式顺序，减少固定先后次序偏差。
所有 mode 使用同一份模型与 prompt；每次生成建立新的缓存。
摘要报告 trial 总耗时、TTFT、吞吐的中位数，以及汇总 token 步间隔的 p50/p95。
短测量中的 p95 只是描述统计；不应解释为生产尾延迟。

## 内存字段

- `cache_used_bytes`：结束时有效 K/V 元素的逻辑大小。
- `cache_allocated_bytes`：缓存 tensor 容量，不含 dynamic 更新的临时新旧副本峰值。
- `cuda_baseline_allocated_bytes`：计时前 PyTorch 活跃 tensor 内存。
- `cuda_peak_allocated_bytes`：本次调用中的 PyTorch 活跃 tensor 峰值，包括模型和输入。

CUDA caching allocator 的 reserved 内存、驱动内存和整卡 `nvidia-smi` 用量
与这些字段口径不同。CPU 不测进程峰值内存，CUDA 字段以 null 表示。
显存中的 static 预留容量与实际有效长度要分开解释。

## 输出与结论

JSON 保存环境和原始样本，CSV 保存各模式摘要。记录完整命令和 git commit，
在报告里写清 CPU/GPU 型号、PyTorch/CUDA 版本、shape 与精度。
默认 float32，没有自动混合精度、CUDA Graph、融合 attention 或编译优化。

小规模下 Python、检查与同步可能主导耗时；缓存模式也可能比重算慢。
先观察多组负载，再解释原因。真实 GPU 实测之前，不填写预期提速数字。

`examples/` 中的数据如有提供，仅代表其 JSON 标记的那次本机 CPU 测量。
之后在 4090 上执行相同协议，使用新的结果前缀保存，保留所有 trial。
