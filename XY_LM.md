# XY-LM 模型文档 (v2)

本文档详细介绍了 XY-LM 模型的架构、用途、训练数据格式，以及如何进行训练和推理。

## 1. 模型架构

XY-LM 是一个基于 **RWKV-7** 架构的自回归语言模型，专为高质量的文本转语音（TTS）任务设计。其核心创新在于采用了**多通道（Multi-Channel）输入和输出**的统一建模方式，将文本和多层级的声学特征码（audio tokens）融合在同一个模型中进行处理。

### 1.1. 核心设计：多通道融合

- **输入**: 模型的输入是一个三维张量，形状为 `(B, T, C)`，其中 `B` 是批量大小，`T` 是序列长度，`C` 是通道数（本项目中为8）。

- **通道定义**: 
  - **通道 0 (混合通道)**: 这是模型的关键。它并非纯文本通道，而是**混合了文本 Token 和经过偏移的音频通道0 Token**。具体来说，音频 Token 的 ID 会被加上一个巨大的偏移量（`text_shift_size`, 如 65536），以在词汇表中将它们与文本 Token 区分开。
  - **通道 1-7 (纯音频通道)**: 这些通道承载由 `XY_Tokenizer` 生成的原始音频 Token。

- **嵌入层 (Embedding)**:
  - 模型为 8 个通道分别配备了独立的嵌入层 (`nn.ModuleList`)。
  - 在前向传播时，模型获取每个通道的嵌入向量，然后将这 **8 个嵌入向量逐元素相加（Summation）**，形成一个单一的、在每个时间步都融合了文本和所有音频层级信息的向量。这个融合后的向量是 RWKV 主干的最终输入。

- **输出头 (Output Head)**:
  - RWKV-7 主干输出的隐藏状态（hidden_states）被**同时**送入 8 个独立的输出头 (`nn.ModuleList`)，每个头分别预测对应通道的下一个 Token。

## 2. 输入张量构造：一步步解析

理解模型如何将文本和音频混合成一个三维张量是理解整个项目的关键。下面我们通过一个例子来详细说明。

**假设:**
- **文本 Prompt**: "你好"
- **音频 Prompt**: 一段音频，经过 `XY_Tokenizer` 后生成了 3 个时间步的音频 Token，形状为 `(8, 3)`。

```python
# 文本 Token (T_text=2)
text_tokens = [34, 42] # 假设 "你好" 被编码为 [34, 42]

# 音频 Tokens (T_audio=3, C=8)
audio_tokens = [
    [101, 102, 103], # ch0
    [201, 202, 203], # ch1
    [301, 302, 303], # ch2
    ...
    [801, 802, 803]  # ch7
]

# 配置
text_shift_size = 65536
text_pad_id = 0
audio_pad_id = 1023
```

### 步骤 1: 构造混合通道 (通道 0)

我们将文本 Token 和**偏移后**的音频通道0 Token 拼接起来。

```python
shifted_audio_ch0 = [101 + 65536, 102 + 65536, 103 + 65536]
# -> [65637, 65638, 65639]

final_ch0 = text_tokens + shifted_audio_ch0
# -> [34, 42, 65637, 65638, 65639]
```

### 步骤 2: 应用时间偏移 (Time Shifting)

这是模型能够学习音频内部结构的核心机制。如果我们将所有8个通道的音频 Token 在同一时间步输入，模型将难以学习它们之间的因果关系（例如，如何根据 ch0-ch6 的信息来预测 ch7）。

**因此，我们将音频通道在时间维度上进行“阶梯式”的排列。** 通道 `c` 的输入会比通道 `c-1` 延迟一个时间步。这创造了一种跨通道的自回归依赖关系。

最终的 `input_ids` 长度会是 `T_text + T_audio + (C-1)`。

### 步骤 3: 组装最终的 `input_ids` 张量

结合上述逻辑，我们来构建最终的 `(T, C)` 张量。`T = 2 + 3 + (8-1) = 12`。

| Time | ch0 (混合) | ch1 (音频) | ch2 (音频) | ch3 (音频) | ... | ch7 (音频) |
|:----:|:----------:|:----------:|:----------:|:----------:|:---:|:----------:|
| 0    | 34         | `audio_pad`| `audio_pad`| `audio_pad`| ... | `audio_pad`|
| 1    | 42         | `audio_pad`| `audio_pad`| `audio_pad`| ... | `audio_pad`|
| 2    | **65637**  | `audio_pad`| `audio_pad`| `audio_pad`| ... | `audio_pad`|
| 3    | **65638**  | **201**    | `audio_pad`| `audio_pad`| ... | `audio_pad`|
| 4    | **65639**  | **202**    | **301**    | `audio_pad`| ... | `audio_pad`|
| 5    | `text_pad` | **203**    | **302**    | **401**    | ... | `audio_pad`|
| 6    | `text_pad` | `audio_pad`| **303**    | **402**    | ... | `audio_pad`|
| ...  | ...        | ...        | ...        | ...        | ... | ...        |
| 11   | `text_pad` | `audio_pad`| `audio_pad`| `audio_pad`| ... | **803**    |

观察上表，你可以清晰地看到音频 Token 是如何沿着对角线传播的。这使得模型在时间步 `t` 预测时，能够利用 `t-1` 的所有通道信息、`t-2` 的所有通道信息等，从而建立起复杂的时序和跨通道依赖关系。

### 步骤 4: 生成最终输入嵌入 (Embedding Generation)

得到 `input_ids` (形状 `B, T, C`) 后，模型执行以下操作：

1.  **独立嵌入**: 对每个通道 `c`，使用其专属的嵌入层 `self.embs[c]` 将 `input_ids[:, :, c]` 转换为嵌入向量 `embed_c` (形状 `B, T, H`)。
2.  **堆叠**: 将 8 个通道的嵌入向量堆叠起来，形成一个 `(C, B, T, H)` 的张量。
3.  **求和**: 沿着通道维度 `dim=0` 进行求和，将 8 个向量融合成一个最终的输入嵌入 `inputs_embeds` (形状 `B, T, H`)。

这个融合后的向量，既包含了文本的语义信息，也包含了所有声学层级的时序信息，是模型强大能力的根基。

## 3. 如何训练

模型使用 `deepspeed` 进行大规模分布式训练。

### 3.1. 训练命令示例

```bash
deepspeed --num_nodes 1 --num_gpus 4 train_scripts/train_xy_llm.py \
  --webdataset_dir /path/to/your/voxbox/ \
  --model_name_or_path /path/to/your/rwkv7-xy-0.4B-g1/ \
  --output_dir /path/to/your/output/xylm \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing True \
  --ds_stage 2 \
  --learning_rate 6e-4 \
  --learning_rate_final 3e-4 \
  --xy_tokenizer_config_path third_party/XY_Tokenizer/config/xy_tokenizer_config.yaml \
  --xy_tokenizer_ckpt_path /path/to/your/XY_Tokenizer_TTSD_V0/xy_tokenizer.ckpt \
  --max_tokens_per_round 16
```

### 3.2. 关键参数说明

- `--max_tokens_per_round`: **动态批量切片**。限制每轮训练的总 Token 数（`batch_size * sequence_length`），防止因序列过长导致显存溢出。脚本会自动将超长的批次切片到安全尺寸。

## 4. 如何预测

推理过程通过自定义的 `.generate()` 函数实现，其核心是处理好多通道的生成和停止逻辑。

### 4.1. 生成逻辑

1.  **Prefill**: 模型处理输入的文本和/或音频 prompt，填充初始 KV 缓存。
2.  **自回归生成**: 
    - **通道0约束**: 在生成音频阶段，通道0的 logits 被严格限制在 `[65536, 65536 + 1024)` 的音频 Token 范围内，确保不会生成文本。
    - **停止与刷新**: 当模型（通常是通道0）需要停止时（例如，由 `stopping_criteria` 触发），会启动一个 `C-1` (即7)步的“刷新”倒计时。在此期间，模型会强制在通道0输出 `EOS`，并在其他音频通道上输出 `PAD`，以优雅地结束所有时间偏移的音频流。

### 4.2. 推理代码示例 (伪代码)

```python
from transformers import AutoTokenizer
from model.llm.xy_llm import RWKV7XYLM

# 1. 加载模型和分词器
model = RWKV7XYLM.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 准备输入 (需要遵循上述的“输入张量构造”逻辑)
# input_ids.shape -> (1, T, 8)

# 3. 调用生成函数
output_sequences = model.generate(input_ids=input_ids, max_new_tokens=512)

# 4. 解码输出
# 从 output_sequences 中提取新生成的、未偏移的语音 Token
# 然后使用 XY_Tokenizer 将其解码回音频波形
```