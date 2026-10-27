# KV Cache：保存了什么，省下了什么

## Prefill 与 decode

对于本项目的 causal decoder，历史 token 不会关注未来 token。
当模型权重、位置和历史前缀不变时，历史 token 在各层的 K/V 可以重复使用。
缓存按层保存，shape 是 `[B, H, T_cached, head_dim]`。

假设 prompt 有 4 个 token：

| 时刻 | 新输入长度 | 该层 attention 里的 K/V 长度 | 本轮选出什么 |
|---|---:|---:|---|
| Prefill | 4 | 4 | 第 1 个输出 token |
| Decode 1 | 1 | 5 | 第 2 个输出 token |
| Decode 2 | 1 | 6 | 第 3 个输出 token |

生成 N 个 token 一共执行 N 次模型调用，其中 1 次 prefill、N-1 次 decode。
最后选出的 token 还没被再次送入模型，所以结束时缓存有效长度是
`prompt_length + N - 1`。static 模式为整个请求声明的
`prompt_length + N` 上限预留容量，保留一个尚未写入的位置。

## Dynamic 与 static

dynamic 每次用 `torch.cat` 生成新 K/V tensor。它按有效长度占用缓存，
但增长时需要复制历史数据，并在瞬间同时持有旧缓存与新缓存。

static 第一次更新每层时分配整段容量，再使用 `copy_` 写入对应切片。
它用预留空间换取稳定的缓存存储地址。返回值只包含已写入的前缀，
未初始化的位置永远不参与 attention。

`used_bytes` 统计有效 token 的逻辑字节；`allocated_bytes` 统计缓存 tensor
自身的容量字节。两者都不能代替整个程序的峰值内存。

## 两个很容易写错的地方

**位置偏移。** 有缓存后，新 token 的位置从 `past_length` 开始，
不能每次都重新从 0 开始。否则 shape 看着正常，计算结果却已经改变。

**带历史的因果 mask。** 如果缓存已有 4 个 token，本轮输入 3 个新 token，
查询位置是 `[4, 5, 6]`，键位置是 `[0, 1, 2, 3, 4, 5, 6]`。
允许关注的条件为 `key_position <= query_position`。
直接对一个 3×7 矩阵使用从左上角开始的普通下三角 mask 会屏蔽合法历史。

测试将全序列输入拆成多种 chunk，并逐位置比较 logits，专门覆盖这个问题。

## 使用约定

- 使用 `model.eval()` 和 `torch.inference_mode()` 执行缓存推理。
- 一个缓存属于同一个模型的一次请求或同长度 batch；新请求创建新缓存。
- 序列继续增长时 batch、heads、head_dim、dtype 和 device 保持不变。
- 本版本不支持 padding、变长 batch、beam search、缓存分叉或跨模型共享。

原理参考：[Hugging Face 缓存说明](https://huggingface.co/docs/transformers/cache_explanation)。
