# Tar音频Token提取工具

这是一个用于从tar格式的音频数据集中提取音频token的工具。该工具支持多进程并行处理，能够高效地从大量音频文件中提取BiCodec音频token。

> **注意**: 这是 `utils/extract_tar_tokens.py` 工具的专用说明文档。

## 功能特性

- 🚀 **多进程并行处理**: 支持多GPU并行处理，提高处理效率
- 🎵 **音频格式支持**: 支持多种音频格式的自动转换和重采样
- 💾 **内存优化**: 内置内存管理和垃圾回收机制
- 📊 **实时监控**: 提供处理进度和性能统计信息
- 🔧 **灵活配置**: 支持自定义处理参数和GPU分配策略

## 环境要求

- Python 3.7+
- PyTorch
- torchaudio
- soundfile
- datasets
- librosa
- psutil
- tqdm

## 安装依赖

```bash
pip install torch torchaudio soundfile datasets librosa psutil tqdm
```

## 模型下载

在使用工具之前，需要下载Spark-TTS模型：

### Spark-TTS-0.5B模型
- **下载地址**: [Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B)
- **模型大小**: 约500MB
- **下载方式**: 从ModelScope下载并解压到指定目录

```bash
# 示例：下载并设置模型目录
mkdir -p /home/yueyulin/models/
# 从ModelScope下载Spark-TTS-0.5B模型到 /home/yueyulin/models/Spark-TTS-0.5B/
```

## 使用方法

### 基本用法

```bash
python utils/extract_tar_tokens.py \
    --input_dir /path/to/tar/files \
    --output_dir /path/to/output \
    --model_dir /path/to/spark-tts/model
```

### 完整参数说明

```bash
python utils/extract_tar_tokens.py \
    --input_dir /tmp/tmp_data/ \
    --output_dir /home/yueyulin/data/Emilia/ZH/tar_tokens/ \
    --model_dir /home/yueyulin/models/Spark-TTS-0.5B/ \
    --num_proc 4 \
    --from_index 0 \
    --to_index 100
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | str | `/tmp/tmp_data/` | 输入tar文件目录路径 |
| `--output_dir` | str | `/home/yueyulin/data/Emilia/ZH/tar_tokens/` | 输出JSONL文件目录路径 |
| `--model_dir` | str | `/home/yueyulin/models/Spark-TTS-0.5B/` | Spark-TTS模型目录路径，模型下载地址：[Spark-TTS-0.5B](https://modelscope.cn/models/SparkAudio/Spark-TTS-0.5B) |
| `--num_proc` | int | 4 | 并行处理进程数量 |
| `--from_index` | int | 0 | 开始处理的文件索引（包含） |
| `--to_index` | int | None | 结束处理的文件索引（不包含），None表示处理所有文件 |

## 输出格式

工具会为每个进程生成一个JSONL文件，文件名格式为：`{input_dir_name}_{process_id}.jsonl`

每个输出行包含以下字段：

```json
{
    "language": "zh",
    "text": "音频对应的文本内容",
    "global_tokens": [1, 2, 3, ...],
    "semantic_tokens": [1, 2, 3, ...]
}
```

## GPU分配策略

工具会根据进程ID自动分配GPU资源：
- 每个GPU最多分配2个进程
- GPU分配公式：`gpu_id = (process_id // 2) % device_count`
- 如果没有可用GPU，将使用CPU处理

### 大音频文件处理

**重要提示**: 如果处理特别大的音频文件（如长音频、高采样率音频），请修改GPU分配策略：

1. **修改GPU分配公式**: 将每个GPU的进程数限制为1个
   ```python
   # 在代码中修改 get_available_gpu 函数
   gpu_id = process_id % device_count  # 每个GPU只分配1个进程
   ```

2. **设置进程数**: 将 `--num_proc` 参数设置为1
   ```bash
   python utils/extract_tar_tokens.py \
       --input_dir /path/to/large/audio/files \
       --output_dir /path/to/output \
       --model_dir /path/to/spark-tts/model \
       --num_proc 1  # 大音频文件建议使用单进程
   ```

**原因说明**:
- 大音频文件会占用更多GPU内存
- 多进程并行处理可能导致GPU内存不足
- 单进程处理可以避免内存竞争和OOM错误

## 性能优化

### 内存管理
- 自动清理GPU缓存和内存
- 每处理1000个请求输出统计信息
- 内存使用超过100GB时触发额外清理

### 处理统计
工具会定期输出处理统计信息：
- 总请求数
- 总处理时间
- 平均处理时间

## 错误处理

- 自动处理音频格式转换错误
- 支持采样率自动重采样
- 进程异常时自动重启
- 支持键盘中断优雅退出

## 使用示例

### 处理指定范围的文件

```bash
# 处理第10到第50个tar文件
python utils/extract_tar_tokens.py \
    --input_dir /data/audio_tars \
    --output_dir /output/tokens \
    --from_index 10 \
    --to_index 50 \
    --num_proc 8
```

### 使用更多进程提高处理速度

```bash
# 使用8个进程并行处理
python utils/extract_tar_tokens.py \
    --input_dir /data/audio_tars \
    --output_dir /output/tokens \
    --num_proc 8
```

## 注意事项

1. **内存使用**: 工具会占用较多内存，建议在内存充足的机器上运行
2. **GPU资源**: 确保有足够的GPU内存，每个进程会加载完整的BiCodec模型
3. **磁盘空间**: 确保输出目录有足够的磁盘空间
4. **文件格式**: 输入文件必须是tar格式，包含音频和JSON元数据

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`--num_proc`参数值
   - 确保GPU内存充足

2. **进程初始化失败**
   - 检查模型路径是否正确
   - 确认模型文件完整性

3. **输出文件为空**
   - 检查输入tar文件格式
   - 确认音频数据完整性

## 许可证

请参考项目主许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

---

**相关文件**: `utils/extract_tar_tokens.py`
