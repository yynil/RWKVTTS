import os
import tempfile
import torch
import torchaudio
import gradio as gr
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 全局变量
model_path = '/external_data/models/CosyVoice2-0.5B_RWKV_0.19B/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 在应用启动时初始化模型（全局共享）
print("正在初始化 CosyVoice2 模型...")
cosyvoice = CosyVoice2(model_path, device=device, fp16=True)
# 预热模型
cosyvoice.model.llm.dummy_forward()
print("模型初始化完成！")

def synthesize_speech(audio_file, prompt_text, tts_text):
    """合成语音"""
    global cosyvoice
    
    if not audio_file or not prompt_text or not tts_text:
        return None, "请提供所有必需的输入（提示音频、提示文本和要合成的文本）"
    
    try:
        # 加载提示音频
        prompt_speech_16k = load_wav(audio_file, 16000)
        
        # 执行推理
        result = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False)
        
        # 获取合成的语音
        output_speech = result[0]['tts_speech']
        
        # 保存临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        torchaudio.save(temp_file.name, output_speech, cosyvoice.sample_rate)
        
        return temp_file.name, f"语音合成成功！"
    except Exception as e:
        return None, f"合成过程中出错：{str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="RWKV TTS 演示") as demo:
    gr.Markdown("# RWKV 语音合成演示")
    gr.Markdown("### 语音合成系统已准备就绪，可直接使用")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="上传提示音频文件（WAV 格式）")
            prompt_text = gr.Textbox(label="提示文本（与提示音频对应的文字内容）", placeholder="例如：今天天气挺不错的。")
            tts_text = gr.Textbox(label="要合成的文本", placeholder="例如：收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。")
            synthesize_button = gr.Button("生成语音")
        
        with gr.Column():
            audio_output = gr.Audio(label="合成的语音")
            output_message = gr.Textbox(label="状态信息")
    
    synthesize_button.click(
        fn=synthesize_speech,
        inputs=[audio_input, prompt_text, tts_text],
        outputs=[audio_output, output_message]
    )
    
    gr.Markdown("""
    ## 使用说明
    
    1. 上传一个WAV格式的提示音频文件
    2. 输入与提示音频对应的文本内容
    3. 输入希望合成的文本
    4. 点击"生成语音"按钮进行语音合成
    
    注意：模型已在服务启动时预加载，所有用户共享同一个模型实例。
    """)

# 启动应用
if __name__ == "__main__":
    demo.launch()