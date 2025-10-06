#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RWKV ASR Gradio 5.x 录音应用
使用最新的Gradio 5.x语法
"""

import os
import sys
import time
from pathlib import Path
import click
import torch
import gradio as gr

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.asr_inference_with_chatrwkv import (
    load_asr_models, 
    inference_asr, 
    AsrModels
)

class Gradio5ASR:
    def __init__(self, audio_lm_path: str, llm_path: str, whisper_path: str, 
                 tokenizer_path: str, device: str, dtype: str):
        """初始化Gradio 5.x ASR应用"""
        self.audio_lm_path = audio_lm_path
        self.llm_path = llm_path
        self.whisper_path = whisper_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.dtype = dtype
        
        # 数据类型映射
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }
        self.torch_dtype = dtype_map[dtype]
        
        # 模型实例
        self.models: AsrModels = None
        
        # 创建界面
        self.interface = self.create_interface()
    
    def load_models(self):
        """加载ASR模型"""
        if self.models is None:
            print("正在加载ASR模型...")
            self.models = load_asr_models(
                self.audio_lm_path, 
                self.llm_path, 
                self.whisper_path, 
                self.tokenizer_path, 
                self.device, 
                self.torch_dtype
            )
            print("ASR模型加载完成")
    
    def process_audio(self, audio_path: str, language: str) -> str:
        """处理音频文件"""
        if audio_path is None:
            return "请先录音"
        
        try:
            # 确保模型已加载
            self.load_models()
            
            # 执行ASR推理
            print(f"开始处理音频: {audio_path}, 语言: {language}")
            start_time = time.time()
            
            results = inference_asr(self.models, audio_path, language, self.torch_dtype, self.device)
            decoded_text = self.models.tokenizer.decode(results)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"识别完成，耗时: {processing_time:.2f}秒")
            return f"识别结果: {decoded_text}\n\n处理时间: {processing_time:.2f}秒"
            
        except Exception as e:
            error_msg = f"识别失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    def create_interface(self):
        """创建Gradio 5.x界面"""
        with gr.Blocks(
            title="RWKV ASR 录音识别系统",
            theme=gr.themes.Soft()
        ) as interface:
            
            gr.Markdown("""
            # 🎤 RWKV ASR 录音识别系统
            
            支持中文和英文录音识别，点击录音按钮开始录音。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # 语言选择
                    language = gr.Radio(
                        choices=["chinese", "english"],
                        value="chinese",
                        label="选择识别语言",
                        info="选择语音识别的目标语言"
                    )
                    
                    # 录音组件 - Gradio 5.x最新语法
                    audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="录音",
                        interactive=True
                    )
                    
                    # 识别按钮
                    recognize_btn = gr.Button("🚀 开始识别", variant="primary")
                
                with gr.Column(scale=1):
                    # 结果显示
                    output = gr.Textbox(
                        label="识别结果",
                        lines=15,
                        interactive=False,
                        placeholder="识别结果将显示在这里...",
                        show_copy_button=True
                    )
                    
                    # 状态显示
                    status = gr.Textbox(
                        label="状态",
                        lines=2,
                        interactive=False,
                        value="就绪"
                    )
            
            # 处理函数
            def process_recording(audio_path, language):
                """处理录音"""
                if audio_path is None:
                    return "请先录音", "等待录音"
                
                status_text = f"正在处理录音... (语言: {language})"
                result = self.process_audio(audio_path, language)
                
                if "识别失败" in result:
                    return result, "处理失败"
                else:
                    return result, "处理完成"
            
            # 绑定事件
            recognize_btn.click(
                fn=process_recording,
                inputs=[audio, language],
                outputs=[output, status]
            )
            
            # 添加使用说明
            gr.Markdown("""
            ## 📖 使用说明
            
            1. **选择语言**: 选择中文或English
            2. **开始录音**: 点击录音按钮，允许浏览器访问麦克风
            3. **说话**: 对着麦克风清晰说话
            4. **停止录音**: 录音会自动停止
            5. **开始识别**: 点击"开始识别"按钮
            6. **查看结果**: 识别结果将显示在右侧
            
            ## 💡 录音技巧
            
            - 保持安静的环境
            - 距离麦克风适中（10-20cm）
            - 说话清晰，语速适中
            - 避免背景噪音
            """)
        
        return interface
    
    def launch(self, host='0.0.0.0', port=7860, share=False, debug=False):
        """启动Gradio 5.x应用"""
        print(f"启动RWKV ASR 录音识别系统 (Gradio 5.x)...")
        print(f"访问地址: http://{host}:{port}")
        
        self.interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug
        )

@click.command()
@click.option('--audio-lm-path', default="/home/yueyulin/models/rwkv7_0.1b_audio_lm_latents_1.5b_202k", 
              help='音频语言模型路径')
@click.option('--llm-path', default="/home/yueyulin/models/rwkv7-g1a-1.5b-20250922-ctx4096.pth", 
              help='大语言模型路径')
@click.option('--whisper-path', default="/home/yueyulin/models/whisper-large-v3/", 
              help='Whisper模型路径')
@click.option('--tokenizer-path', default="tokenizer/rwkv_vocab_v20230424.txt", 
              help='分词器路径')
@click.option('--device', default="cuda:0", 
              help='设备类型 (cuda:0/cpu)')
@click.option('--dtype', default="float16", 
              type=click.Choice(['float16', 'float32', 'bfloat16']),
              help='数据类型')
@click.option('--host', default="0.0.0.0", 
              help='服务器主机地址')
@click.option('--port', default=7860, 
              help='服务器端口')
@click.option('--share', is_flag=True, 
              help='创建公共链接')
@click.option('--debug', is_flag=True, 
              help='启用调试模式')
def main(audio_lm_path, llm_path, whisper_path, tokenizer_path, device, dtype, host, port, share, debug):
    """
    RWKV ASR Gradio 5.x 录音应用主程序
    """
    # 检查模型路径
    if not os.path.exists(audio_lm_path):
        print(f"错误: 音频语言模型路径不存在: {audio_lm_path}")
        return
    
    if not os.path.exists(llm_path):
        print(f"错误: 大语言模型路径不存在: {llm_path}")
        return
    
    if not os.path.exists(whisper_path):
        print(f"错误: Whisper模型路径不存在: {whisper_path}")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"错误: 分词器路径不存在: {tokenizer_path}")
        return
    
    # 创建并启动Gradio应用
    app = Gradio5ASR(
        audio_lm_path=audio_lm_path,
        llm_path=llm_path,
        whisper_path=whisper_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype
    )
    
    app.launch(host=host, port=port, share=share, debug=debug)

if __name__ == "__main__":
    main()
