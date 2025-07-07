# ReSpark 语音生成测试

这个目录包含了用于测试 ReSpark 模型语音生成功能的代码。

## 文件说明

### 1. `test_respark_model.py`
完整的测试脚本，包含详细的生成函数和命令行参数支持。

### 2. `simple_generate.py`
简化的生成脚本，用于快速测试和演示。

## 重要参数说明

### EOS Token ID
根据训练代码分析，`eos_token_id` 设置为：
```python
eos_token_id = model.config.vocab_size - 1
```

这个设置基于以下训练代码：
- `train_spark_rwkv7speech_multiple_dataset.py` 第515行
- `train_spark_rwkv7speech.py` 第521行
- `train_spark_rwkv7speech_jsonl.py` 第470行

### 生成参数
- `max_new_tokens`: 最大生成token数（默认3000）
- `do_sample`: 是否使用采样（默认True）
- `top_k`: top-k采样参数（默认50）
- `top_p`: top-p采样参数（默认0.95）
- `temperature`: 温度参数（默认1.0）

## 使用方法

### 方法1：使用完整测试脚本

```bash
python test_respark_model.py \
    --model_dir /path/to/your/model \
    --spark_model_dir /path/to/spark/model \
    --prompt_audio_path /path/to/prompt_audio.wav \
    --text "要生成的文本" \
    --output_path output.wav \
    --max_new_tokens 1000 \
    --temperature 0.8
```

### 方法2：使用简化脚本

1. 编辑 `simple_generate.py` 中的路径配置：
```python
model_dir = "/path/to/your/model/"
spark_model_dir = "/path/to/spark/model/"
prompt_audio_path = "/path/to/prompt_audio.wav"
```

2. 运行脚本：
```bash
python simple_generate.py
```

## 生成流程

1. **加载模型和分词器**
   - 加载 ReSpark 语言模型
   - 加载文本分词器
   - 加载 BiCodecTokenizer

2. **处理提示音频**
   - 加载提示音频文件
   - 重采样到16kHz（如果需要）
   - 提取 global_tokens 和 semantic_tokens

3. **生成输入嵌入**
   - 使用 `generate_embeddings` 函数
   - 组合文本、global_tokens 和 semantic_tokens

4. **模型生成**
   - 使用 `model.generate()` 方法
   - 设置 `eos_token_id = model.config.vocab_size - 1`
   - 应用采样参数

5. **处理生成的Tokens**
   - **重要**：直接使用生成的token ID，不进行文本解码
   - 生成的token ID是模型词表中的ID，不是原始tokenizer的词表
   - 这些token ID直接作为semantic tokens使用

6. **音频解码**
   - 使用 BiCodec 解码器
   - 将 semantic_tokens 和 global_tokens 转换为音频

## 关键理解

### Token处理方式
**重要修正**：生成的token ID应该直接使用，而不是尝试解码为文本。

```python
# 错误的方式（之前的方法）
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
semantic_tokens = re.findall(r"bicodec_semantic_(\d+)", generated_text)

# 正确的方式（现在的方法）
semantic_tokens_tensor = generated_tokens.unsqueeze(0).to(device)
```

### 原因
1. **模型结构**：`RWKV7ForSpeech` 有自己的 `lm_head`，输出维度是 `config.vocab_size`
2. **词表不共享**：生成的token ID是模型自己的词表中的ID，不是原始tokenizer的词表
3. **直接映射**：生成的token ID直接对应BiCodec的semantic token ID

## 注意事项

1. **提示音频必需**：生成过程需要提示音频来提取说话人的声音特征（global_tokens）

2. **采样率要求**：BiCodecTokenizer 期望16kHz采样率的音频

3. **EOS Token**：确保正确设置 `eos_token_id`，这对于停止生成至关重要

4. **内存使用**：生成过程可能需要大量GPU内存，请确保有足够的资源

5. **Token处理**：直接使用生成的token ID，不要尝试解码为文本

## 故障排除

### 常见问题

1. **生成失败或音频质量差**
   - 检查模型是否正确训练
   - 调整生成参数（temperature、top_k、top_p）
   - 增加 `max_new_tokens`
   - 确保使用正确的token处理方式

2. **内存不足**
   - 减少 `max_new_tokens`
   - 使用较小的模型
   - 启用梯度检查点

3. **音频质量差**
   - 调整采样参数
   - 使用更好的提示音频
   - 检查模型训练质量

### 调试建议

1. 启用详细日志输出
2. 检查生成的token ID序列
3. 验证输入嵌入的形状和内容
4. 确认EOS token ID的正确性
5. 确保直接使用生成的token ID而不是解码 