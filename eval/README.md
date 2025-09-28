# TTS 评测

1. 评测代码：https://github.com/BytedanceSpeech/seed-tts-eval
 评测 WER，生成错误率和 SIM 模拟相似度。
 评测数据下载：https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit

2. 当前 ReSpark 实现：
 eval_tts_base.py ResparkTTSEngine
 下载模型最新的 checkpoint：
 LLM：https://huggingface.co/yueyulin/respark/tree/main/rwkv7-0.4B-g1-respark-voice-tunable
 SparkTTS（需要用到 BiCodec）：https://huggingface.co/SparkAudio/Spark-TTS-0.5B/tree/main
 这些在/home/yueyulin/models 下面都有，这个目录权限是 777， 请不要误删数据。seed-tts-eval 的数据也在这个目录下面。

3. WER评测逻辑：
 3.1 用 eval_seed_generate.py 合成语音
 3.2 用 run_wer.py 计算错误、删、改、插比例
 3.3 最后用https://github.com/BytedanceSpeech/seed-tts-eval/blob/main/average_wer.py 计算整体 WER

4. SIM 评测逻辑类似。需完善。 
