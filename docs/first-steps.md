# 第一次读源码：从 shape 到一个新 token

## 先运行，再沿着数据走

```bash
python -m decodelab trace --dim 32 --heads 4 --layers 1 --prompt 1,2,3
```

这个命令使用随机初始化的小模型，作用是展示计算。输入数字本身就是 token ID，
暂时把分词器留在模型外。先关注 PREFILL 中的五个位置：

1. `input_ids` 是 `[1, 3]`：一条请求、三个 token。
2. `embedding` 是 `[1, 3, 32]`：每个 ID 变成一个 32 维向量，加上位置向量。
3. `query` 是 `[1, 4, 3, 8]`：四个 attention head，每个 head 的维度是 8。
4. `attention_scores` 是 `[1, 4, 3, 3]`：每个查询位置给各个键位置打分。
5. `logits` 是 `[1, 3, 256]`：每个输入位置对应一组词表候选分数。

生成下一 token 时取最后一个输入位置的分数，因此使用 `logits[:, -1]`。
`argmax` 选出得分最高的编号；这就是本项目的 greedy 策略。
当模型未经训练时，这个编号只用于计算验证。

## 模型代码只跟三层调用

打开 `src/decodelab/model.py`，按下列顺序阅读：

- `TinyDecoder.forward`：输入检查、embedding、遍历 block、输出 logits。
- `DecoderBlock.forward`：归一化、attention、残差连接、MLP。
- `CausalAttention.forward`：QKV 投影、分 head、历史缓存、mask、softmax、加权求和。

看到 `reshape` 就手写前后 shape；看到 `transpose` 就标记交换了哪两个维度；
看到矩阵乘法就对照最后两个维度。先拿 `B=1, T=3, D=32, H=4` 跟到底。

`scores` 的最后一维对应“可以关注哪些 token”。对这个维度做 softmax 后，
每个查询位置得到各个历史位置的权重，再用它们对 V 加权求和。
causal mask 会把未来位置屏蔽掉。

## 再看模型外的循环

打开 `src/decodelab/generation.py`，找到 `for step in range(max_new_tokens)`：

1. 准备这一轮送入模型的 token。
2. 执行 `model(inputs, cache=cache)`。
3. 从最后位置的 logits 中选下一个 ID。
4. 把新 ID 追加到输出序列。
5. 记录这个 token 准备完成的时间。

`none` 模式每轮把整段序列再次送入模型。
有缓存时，第一轮仍处理完整 prompt，之后只送上轮刚选出的一个 token。

## 建议亲手完成的三个练习

- 把 prompt 从 3 个 token 改成 5 个，先预测各层 shape，再运行验证。
- 观察 DECODE：Q 只有一个位置，K 却有历史位置；解释 attention 分数为何不是方阵。
- 修改一个序列未来位置的 token，检查前面位置的 logits 是否保持不变；
  阅读 `tests/test_inference.py` 中的 causal 测试理解原因。

能解释这些现象后，再读 `cache.py`。PyTorch 基础可配合
[Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
和 [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)。
