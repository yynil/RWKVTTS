import customtkinter as ctk
import threading
import time
import os
import pygame
import soundfile as sf
import numpy as np

# 设置主题
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# 尝试导入TTS模块
try:
    from webrwkv_py import Model, ThreadRuntime, get_available_adapters_py
    from transformers import AutoTokenizer
    import onnxruntime as ort
    HAS_TTS_MODULES = True
    print("✅ TTS模块加载成功")
except ImportError as e:
    print(f"⚠️ 缺少TTS模块: {e}")
    HAS_TTS_MODULES = False

# 导入正确的属性转换函数
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.properties_util import convert_standard_properties_to_tokens
    HAS_PROPERTIES_UTIL = True
    print("✅ 属性工具模块加载成功")
except ImportError as e:
    print(f"⚠️ 缺少属性工具模块: {e}")
    HAS_PROPERTIES_UTIL = False

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=1024)
        self.current_audio = None
        self.is_playing = False
        self.is_paused = False
        self.audio_length = 0
        self.current_position = 0
        self.start_time = 0
        
    def load_audio(self, audio_path: str) -> bool:
        try:
            if os.path.exists(audio_path):
                audio_data, sample_rate = sf.read(audio_path)
                self.audio_length = len(audio_data) / sample_rate
                self.current_audio = audio_path
                self.current_position = 0
                self.is_playing = False
                self.is_paused = False
                return True
            return False
        except Exception as e:
            print(f"加载音频失败: {e}")
            return False
    
    def play(self):
        if self.current_audio:
            try:
                if self.is_paused:
                    # 如果是暂停状态，恢复播放
                    pygame.mixer.music.unpause()
                    self.is_paused = False
                    self.is_playing = True
                    self.start_time = time.time() - self.current_position
                else:
                    # 重新开始播放
                    pygame.mixer.music.load(self.current_audio)
                    pygame.mixer.music.play()
                    self.is_playing = True
                    self.is_paused = False
                    self.start_time = time.time()
                    self.current_position = 0
            except Exception as e:
                print(f"播放失败: {e}")
    
    def pause(self):
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.current_position = time.time() - self.start_time
    
    def resume(self):
        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
            self.start_time = time.time() - self.current_position
    
    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
    
    def get_current_time(self) -> float:
        if self.is_playing and not self.is_paused:
            return time.time() - self.start_time
        return self.current_position
    
    def set_position(self, position: float):
        if self.current_audio and 0 <= position <= self.audio_length:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.current_audio)
                pygame.mixer.music.play(start=position)
                self.start_time = time.time() - position
                self.current_position = position
                self.is_playing = True
                self.is_paused = False
            except Exception as e:
                print(f"设置位置失败: {e}")
    
    def is_audio_ended(self) -> bool:
        """检查音频是否播放结束"""
        if not self.is_playing:
            return False
        current_time = self.get_current_time()
        return current_time >= self.audio_length

class TTSGenerator:
    def __init__(self):
        self.model = None
        self.runtime = None
        self.tokenizer = None
        self.ort_session = None
        self.available_adapters = []
        
    def get_available_adapters(self) -> list:
        try:
            if HAS_TTS_MODULES:
                adapters = get_available_adapters_py()
                # 处理返回的(索引, 设备名)元组列表
                if isinstance(adapters, list) and len(adapters) > 0:
                    if isinstance(adapters[0], tuple) and len(adapters[0]) == 2:
                        # 保存原始适配器列表
                        self.available_adapters = adapters
                        # 提取设备名，格式为 "索引: 设备名"
                        return [f"{idx}: {name}" for idx, name in adapters]
                    else:
                        self.available_adapters = adapters
                        return list(adapters)
                elif isinstance(adapters, tuple):
                    self.available_adapters = [adapters]
                    return list(adapters)
                else:
                    self.available_adapters = [adapters]
                    return [str(adapters)]
            else:
                return ["CPU (模拟)", "Metal (模拟)", "CUDA (模拟)"]
        except Exception as e:
            print(f"获取设备列表失败: {e}")
            return ["CPU (模拟)", "Metal (模拟)", "CUDA (模拟)"]
    
    def load_model(self, model_path: str, adapter_index: int, progress_callback=None) -> bool:
        """加载TTS模型"""
        try:
            if not HAS_TTS_MODULES:
                print("⚠️ 使用模拟模式加载模型")
                if progress_callback:
                    progress_callback("正在模拟加载模型...", 0.3)
                    time.sleep(1)
                    progress_callback("模型加载完成（模拟）", 1.0)
                return True
            
            print(f"🚀 开始加载模型: {model_path}")
            print(f"📱 选择设备索引: {adapter_index}")
            
            if progress_callback:
                progress_callback("正在加载模型...", 0.2)
            
            # 检查模型文件
            webrwkv_model_path = os.path.join(model_path, 'webrwkv.safetensors')
            if not os.path.exists(webrwkv_model_path):
                raise FileNotFoundError(f"模型文件不存在: {webrwkv_model_path}")
            
            print(f"✅ 找到模型文件: {webrwkv_model_path}")
            
            if progress_callback:
                progress_callback("正在初始化模型...", 0.4)
            
            # 加载模型
            print("🔧 正在初始化WebRWKV模型...")
            precision = 'fp32'
            self.model = Model(webrwkv_model_path, precision, adapter_index)
            print(f"✅ WebRWKV模型初始化成功，精度: {precision}")
            
            if progress_callback:
                progress_callback("正在创建运行时...", 0.6)
            
            # 创建运行时
            print("⚡ 正在创建模型运行时...")
            self.runtime = self.model.create_thread_runtime()
            print("✅ 模型运行时创建成功")
            
            if progress_callback:
                progress_callback("正在加载分词器...", 0.8)
            
            # 加载分词器
            print("📝 正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("✅ 分词器加载成功")
            
            if progress_callback:
                progress_callback("模型加载完成！", 1.0)
            
            print(f"🎉 模型加载完成！路径: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def load_decoder(self, decoder_path: str, progress_callback=None) -> bool:
        """加载音频解码器"""
        try:
            if not HAS_TTS_MODULES:
                print("⚠️ 使用模拟模式加载解码器")
                if progress_callback:
                    progress_callback("正在模拟加载解码器...", 0.5)
                    time.sleep(0.5)
                    progress_callback("解码器加载完成（模拟）", 1.0)
                return True
            
            print(f"🎵 开始加载音频解码器: {decoder_path}")
            
            if progress_callback:
                progress_callback("正在加载ONNX解码器...", 0.5)
            
            # 检查解码器文件
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"解码器文件不存在: {decoder_path}")
            
            print(f"✅ 找到解码器文件: {decoder_path}")
            
            # 加载ONNX解码器
            print("🔧 正在加载ONNX Runtime解码器...")
            self.ort_session = ort.InferenceSession(decoder_path)
            print("✅ ONNX Runtime解码器加载成功")
            
            if progress_callback:
                progress_callback("解码器加载完成！", 1.0)
            
            print(f"🎉 解码器加载完成！路径: {decoder_path}")
            return True
            
        except Exception as e:
            print(f"❌ 解码器加载失败: {e}")
            return False
    
    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        """从logits中采样token"""
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        
        if isinstance(logits, list):
            logits = np.array(logits)
        
        from scipy import special
        probs = special.softmax(logits, axis=-1)
        top_k = int(top_k)
        
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff_mask = cumulative_probs >= top_p
        if np.any(cutoff_mask):
            cutoff_idx = np.argmax(cutoff_mask)
            cutoff = float(sorted_probs[cutoff_idx])
            probs[probs < cutoff] = 0
        
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), size=1, p=probs)
        return int(out[0])
    
    def generate_tts(self, text: str, age: str, gender: str, emotion: str, 
                     pitch: str, speed: str, progress_callback=None, 
                     speed_callback=None) -> np.ndarray:
        """生成TTS音频"""
        
        if not HAS_TTS_MODULES:
            print("⚠️ 使用模拟模式生成TTS")
            return self.simulate_tts_generation(text, progress_callback, speed_callback)
        
        if not self.runtime or not self.tokenizer or not self.ort_session:
            raise RuntimeError("模型或解码器未加载")
        self.runtime.reset()
        # 检查重置后状态是否有非零值
        print("\n🔍 检查重置后状态是否有非零值:")
        after_reset_has_nonzero = self.runtime.check_state_has_nonzero_values()
        print(f"重置后状态有非零值: {after_reset_has_nonzero}")
        print(f"🎯 开始TTS生成...")
        print(f"📝 输入文本: {text}")
        print(f"👤 语音属性: 年龄={age}, 性别={gender}, 情感={emotion}, 音调={pitch}, 语速={speed}")
        
        # TTS标签
        TTS_TAG_0 = 8193
        TTS_TAG_1 = 8194
        TTS_TAG_2 = 8195
        
        if progress_callback:
            progress_callback("正在处理文本和属性...", 0.1)
        
        # 转换属性为token
        properties_tokens = convert_standard_properties_to_tokens(age, gender, emotion, pitch, speed)
        print(f'🔤 属性文本: {properties_tokens}')
        
        # 编码文本和属性 - 按照原始代码的逻辑
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        text_tokens = [i + 8196 + 4096 for i in text_tokens]
        properties_tokens = self.tokenizer.encode(properties_tokens, add_special_tokens=False)
        properties_tokens = [i + 8196 + 4096 for i in properties_tokens]
        
        print(f'🔢 属性token: {properties_tokens}')
        print(f'🔢 文本token: {text_tokens}')
        
        # 组合所有token
        all_idx = properties_tokens + [TTS_TAG_2] + text_tokens + [TTS_TAG_0]
        
        if progress_callback:
            progress_callback("正在进行Prefill...", 0.2)
        
        # Prefill阶段
        print("💎 开始Prefill阶段...")
        start_time = time.time()
        logits = self.runtime.predict(all_idx)
        end_time = time.time()
        prefill_speed = len(all_idx) / (end_time - start_time)
        print(f'⏱️ Prefill完成: {end_time - start_time:.3f}s, 速度: {prefill_speed:.1f} tokens/s')
        
        if speed_callback:
            speed_callback(f"Prefill: {prefill_speed:.1f} tokens/s")
        
        if progress_callback:
            progress_callback("正在生成全局token...", 0.4)
        
        # 生成全局token - 按照原始代码的逻辑
        print("🌍 开始生成全局token...")
        global_tokens_size = 32
        global_tokens = []
        start_time = time.time()
        for i in range(global_tokens_size):
            sampled_id = self.sample_logits(logits[0:4096], temperature=1.0, top_p=0.95, top_k=20)
            global_tokens.append(sampled_id)
            sampled_id += 8196
            logits = self.runtime.predict_next(sampled_id)
            
            if progress_callback:
                progress_callback(f"正在生成全局token... ({i+1}/{global_tokens_size})", 0.4 + 0.2 * (i+1) / global_tokens_size)
        
        end_time = time.time()
        global_speed = 32 / (end_time - start_time)
        print(f'⏱️ 全局token生成完成: {end_time - start_time:.3f}s, 速度: {global_speed:.1f} tokens/s')
        
        if speed_callback:
            speed_callback(f"全局token: {global_speed:.1f} tokens/s")
        print(f'🎯 生成的全局token: {global_tokens}')
        
        if progress_callback:
            progress_callback("正在生成语义token...", 0.7)
        
        # 生成语义token - 按照原始代码的逻辑
        print("🧠 开始生成语义token...")
        x = self.runtime.predict_next(TTS_TAG_1)
        semantic_tokens = []
        start_time = time.time()
        for i in range(2048):
            sampled_id = self.sample_logits(x[0:8193], temperature=1.0, top_p=0.95, top_k=80)
            if sampled_id == 8192:
                print(f"🛑 语义token生成结束，遇到结束标记，共生成 {len(semantic_tokens)} 个token")
                break
            semantic_tokens.append(sampled_id)
            x = self.runtime.predict_next(sampled_id)
            
            if progress_callback:
                progress_callback(f"正在生成语义token... ({len(semantic_tokens)})", 0.7 + 0.2 * min(len(semantic_tokens) / 1000, 1.0))
        
        end_time = time.time()
        semantic_speed = len(semantic_tokens) / (end_time - start_time)
        print(f'⏱️ 语义token生成完成: {end_time - start_time:.3f}s, 速度: {semantic_speed:.1f} tokens/s, 共生成 {len(semantic_tokens)} 个token')
        
        if speed_callback:
            speed_callback(f"语义token: {semantic_speed:.1f} tokens/s")
        
        if progress_callback:
            progress_callback("正在解码音频...", 0.9)
        
        # 准备输入数据 - 按照原始代码的逻辑
        print("🔧 准备解码器输入数据...")
        # 全局token需要减8196，转换为numpy数组
        global_tokens = np.array(global_tokens, dtype=np.int64).reshape(1, 1, -1)
        semantic_tokens = np.array(semantic_tokens, dtype=np.int64).reshape(1, -1)
        print(f'🎯 生成的全局token: {global_tokens}')
        print(f'🎯 生成的语义token: {semantic_tokens}')
        print(f'📊 解码器输入形状: global_tokens={global_tokens.shape}, semantic_tokens={semantic_tokens.shape}')
        
        # 使用ONNX解码器生成音频
        print("🎵 开始ONNX解码器推理...")
        start_time = time.time()
        outputs = self.ort_session.run(None, {"global_tokens": global_tokens, "semantic_tokens": semantic_tokens})
        wav_reconstructed = outputs[0].reshape(-1)
        end_time = time.time()
        
        decode_speed = len(wav_reconstructed) / (end_time - start_time)
        print(f'⏱️ 音频解码完成: {end_time - start_time:.3f}s, 速度: {decode_speed:.0f} samples/s, 音频长度: {len(wav_reconstructed)} 采样点')
        
        if speed_callback:
            speed_callback(f"解码: {decode_speed:.0f} samples/s")
        
        if progress_callback:
            progress_callback("音频生成完成！", 1.0)
        
        print("🎉 TTS音频生成完成！")
        return wav_reconstructed
    
    def simulate_tts_generation(self, text: str, progress_callback=None, speed_callback=None) -> np.ndarray:
        """模拟TTS生成（用于测试）"""
        print("🎭 开始模拟TTS生成...")
        print(f"📝 输入文本: {text}")
        
        if progress_callback:
            progress_callback("正在模拟TTS生成...", 0.1)
            time.sleep(0.5)
            progress_callback("正在处理文本...", 0.3)
            time.sleep(0.5)
            progress_callback("正在生成语音...", 0.6)
            time.sleep(1.0)
            progress_callback("正在解码音频...", 0.8)
            time.sleep(0.5)
            progress_callback("音频生成完成！", 1.0)
        
        if speed_callback:
            speed_callback("模拟: 1000 tokens/s")
            time.sleep(0.5)
            speed_callback("模拟: 800 tokens/s")
            time.sleep(0.5)
            speed_callback("模拟: 1200 tokens/s")
        
        # 生成测试音频
        sample_rate = 16000
        duration = max(2.0, len(text) * 0.1)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        print(f"🎵 模拟音频生成完成: 时长={duration:.1f}s, 采样率={sample_rate}Hz")
        return audio_data

class TTSGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 初始化组件
        self.audio_player = AudioPlayer()
        self.tts_generator = TTSGenerator()
        
        # 初始化变量
        self.model_path = ctk.StringVar()
        self.output_path = ctk.StringVar(value="generated_audio.wav")
        self.decoder_path = ctk.StringVar(value="/Volumes/bigdata/models/BiCodecDetokenize.onnx")
        
        # 属性变量
        self.age_var = ctk.StringVar(value="middle-aged")
        self.gender_var = ctk.StringVar(value="female")
        self.emotion_var = ctk.StringVar(value="NEUTRAL")
        self.pitch_var = ctk.StringVar(value="medium_pitch")
        self.speed_var = ctk.StringVar(value="medium")
        
        # 进度相关变量
        self.progress_var = ctk.DoubleVar()
        self.status_var = ctk.StringVar(value="就绪")
        self.tokens_per_sec_var = ctk.StringVar(value="0 tokens/s")
        
        # 设置窗口
        self.title("RWKV TTS GUI")
        self.geometry("900x1000")
        self.resizable(True, True)
        
        # 创建界面
        self.create_widgets()
        
        # 初始化设备列表
        self.init_devices()
        
        # 绑定文本变化事件
        self.text_input.bind("<KeyRelease>", self.on_text_change)
        
    def create_widgets(self):
        # 主标题
        title_label = ctk.CTkLabel(
            self, 
            text="🎵 RWKV TTS 语音合成系统", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # 创建左右两列的主容器
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 左侧控制面板
        left_panel = ctk.CTkFrame(main_container)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # 右侧播放器面板
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # 在左侧面板创建控制组件
        self.create_left_panel(left_panel)
        
        # 在右侧面板创建播放器组件
        self.create_right_panel(right_panel)
        
    def create_left_panel(self, parent):
        """创建左侧控制面板"""
        # 创建设备选择框架
        self.create_device_frame(parent)
        
        # 创建模型选择框架
        self.create_model_frame(parent)
        
        # 创建文本输入框架
        self.create_text_frame(parent)
        
        # 创建属性设置框架
        self.create_properties_frame(parent)
        
        # 创建生成控制框架
        self.create_generation_frame(parent)
        
        # 创建进度显示框架
        self.create_progress_frame(parent)
        
    def create_right_panel(self, parent):
        """创建右侧播放器面板"""
        # 创建音频播放框架
        self.create_audio_player_frame(parent)
        
        # 创建音频信息显示框架
        self.create_audio_info_frame(parent)
        
    def create_device_frame(self, parent):
        device_frame = ctk.CTkFrame(parent)
        device_frame.pack(fill="x", padx=10, pady=5)
        
        device_label = ctk.CTkLabel(device_frame, text="💎 设备选择", font=ctk.CTkFont(size=16, weight="bold"))
        device_label.pack(pady=10)
        
        self.device_combo = ctk.CTkComboBox(
            device_frame,
            values=["正在加载设备列表..."],
            command=self.on_device_change
        )
        self.device_combo.pack(pady=5)
        
        refresh_btn = ctk.CTkButton(device_frame, text="🔄 刷新设备", command=self.refresh_devices)
        refresh_btn.pack(pady=5)
        
    def create_model_frame(self, parent):
        model_frame = ctk.CTkFrame(parent)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="💎 模型选择", font=ctk.CTkFont(size=16, weight="bold"))
        model_label.pack(pady=10)
        
        # 模型路径输入
        model_path_frame = ctk.CTkFrame(model_frame)
        model_path_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(model_path_frame, text="模型路径:").pack(side="left", padx=5)
        model_entry = ctk.CTkEntry(model_path_frame, textvariable=self.model_path, width=400)
        model_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        browse_btn = ctk.CTkButton(model_path_frame, text="浏览", command=self.browse_model)
        browse_btn.pack(side="right", padx=5)
        
        # 解码器路径输入
        decoder_path_frame = ctk.CTkFrame(model_frame)
        decoder_path_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(decoder_path_frame, text="解码器路径:").pack(side="left", padx=5)
        decoder_entry = ctk.CTkEntry(decoder_path_frame, textvariable=self.decoder_path, width=400)
        decoder_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        decoder_browse_btn = ctk.CTkButton(decoder_path_frame, text="浏览", command=self.browse_decoder)
        decoder_browse_btn.pack(side="right", padx=5)
        
        # 模型加载按钮
        self.load_model_btn = ctk.CTkButton(
            model_path_frame, 
            text="📥 加载模型", 
            command=self.load_model,
            height=30
        )
        self.load_model_btn.pack(side="right", padx=5)
        
    def create_text_frame(self, parent):
        text_frame = ctk.CTkFrame(parent)
        text_frame.pack(fill="x", padx=10, pady=5)
        
        text_label = ctk.CTkLabel(text_frame, text="💎 文本输入", font=ctk.CTkFont(size=16, weight="bold"))
        text_label.pack(pady=10)
        
        self.text_input = ctk.CTkTextbox(text_frame, height=100)
        self.text_input.pack(fill="x", padx=10, pady=5)
        
        # 语言检测显示
        self.lang_label = ctk.CTkLabel(text_frame, text="检测语言: 未输入")
        self.lang_label.pack(pady=5)
        
    def create_properties_frame(self, parent):
        props_frame = ctk.CTkFrame(parent)
        props_frame.pack(fill="x", padx=10, pady=5)
        
        props_label = ctk.CTkLabel(props_frame, text="💎 语音属性", font=ctk.CTkFont(size=16, weight="bold"))
        props_label.pack(pady=10)
        
        props_grid = ctk.CTkFrame(props_frame)
        props_grid.pack(fill="x", padx=10, pady=5)
        
        # 第一行
        row1 = ctk.CTkFrame(props_grid)
        row1.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row1, text="年龄:").pack(side="left", padx=5)
        age_combo = ctk.CTkComboBox(row1, values=["child", "teenager", "youth-adult", "middle-aged", "elderly"], variable=self.age_var)
        age_combo.pack(side="left", padx=5)
        
        ctk.CTkLabel(row1, text="性别:").pack(side="left", padx=5)
        gender_combo = ctk.CTkComboBox(row1, values=["male", "female"], variable=self.gender_var)
        gender_combo.pack(side="left", padx=5)
        
        # 第二行
        row2 = ctk.CTkFrame(props_grid)
        row2.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row2, text="情感:").pack(side="left", padx=5)
        emotion_combo = ctk.CTkComboBox(row2, values=["NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"], variable=self.emotion_var)
        emotion_combo.pack(side="left", padx=5)
        
        ctk.CTkLabel(row2, text="音调:").pack(side="left", padx=5)
        pitch_combo = ctk.CTkComboBox(row2, values=["low_pitch", "medium_pitch", "high_pitch", "very_high_pitch"], variable=self.pitch_var)
        pitch_combo.pack(side="left", padx=5)
        
        # 第三行
        row3 = ctk.CTkFrame(props_grid)
        row3.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row3, text="语速:").pack(side="left", padx=5)
        speed_combo = ctk.CTkComboBox(row3, values=["very_slow", "slow", "medium", "fast", "very_fast"], variable=self.speed_var)
        speed_combo.pack(side="left", padx=5)
        
    def create_generation_frame(self, parent):
        gen_frame = ctk.CTkFrame(parent)
        gen_frame.pack(fill="x", padx=10, pady=5)
        
        gen_label = ctk.CTkLabel(gen_frame, text="💎 生成控制", font=ctk.CTkFont(size=16, weight="bold"))
        gen_label.pack(pady=10)
        
        output_frame = ctk.CTkFrame(gen_frame)
        output_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(output_frame, text="输出路径:").pack(side="left", padx=5)
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_path, width=300)
        output_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        output_browse_btn = ctk.CTkButton(output_frame, text="浏览", command=self.browse_output)
        output_browse_btn.pack(side="right", padx=5)
        
        self.generate_btn = ctk.CTkButton(
            gen_frame, 
            text="🚀 开始生成", 
            command=self.start_generation,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"  # 初始状态为禁用
        )
        self.generate_btn.pack(pady=10)
        
    def create_progress_frame(self, parent):
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        progress_label = ctk.CTkLabel(progress_frame, text="💎 生成进度", font=ctk.CTkFont(size=16, weight="bold"))
        progress_label.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(progress_frame, textvariable=self.status_var, font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.speed_label = ctk.CTkLabel(progress_frame, textvariable=self.tokens_per_sec_var)
        self.speed_label.pack(pady=5)
        
    def create_audio_player_frame(self, parent):
        player_frame = ctk.CTkFrame(parent)
        player_frame.pack(fill="x", padx=10, pady=5)
        
        player_label = ctk.CTkLabel(player_frame, text="🎵 音频播放", font=ctk.CTkFont(size=16, weight="bold"))
        player_label.pack(pady=10)
        
        # 播放控制按钮
        control_frame = ctk.CTkFrame(player_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.play_btn = ctk.CTkButton(control_frame, text="▶️ 播放", command=self.play_audio)
        self.play_btn.pack(side="left", padx=5)
        
        self.pause_btn = ctk.CTkButton(control_frame, text="⏸️ 暂停", command=self.pause_audio)
        self.pause_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(control_frame, text="⏹️ 停止", command=self.stop_audio)
        self.stop_btn.pack(side="left", padx=5)
        
        # 时间显示和进度条
        time_frame = ctk.CTkFrame(player_frame)
        time_frame.pack(fill="x", padx=10, pady=5)
        
        self.current_time_label = ctk.CTkLabel(time_frame, text="00:00")
        self.current_time_label.pack(side="left", padx=5)
        
        self.audio_progress = ctk.CTkSlider(time_frame, from_=0, to=1, command=self.on_audio_progress_change)
        self.audio_progress.pack(side="left", padx=10, fill="x", expand=True)
        
        self.total_time_label = ctk.CTkLabel(time_frame, text="00:00")
        self.total_time_label.pack(side="right", padx=5)
        
        # 音量控制
        volume_frame = ctk.CTkFrame(player_frame)
        volume_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(volume_frame, text="音量:").pack(side="left", padx=5)
        self.volume_slider = ctk.CTkSlider(volume_frame, from_=0, to=1, command=self.on_volume_change)
        self.volume_slider.pack(side="left", padx=10, fill="x", expand=True)
        self.volume_slider.set(0.7)
        
        # 播放状态显示
        status_frame = ctk.CTkFrame(player_frame)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.play_status_label = ctk.CTkLabel(status_frame, text="状态: 就绪")
        self.play_status_label.pack(pady=5)
        
        # 启动定时器更新播放状态
        self.update_playback_timer()
        
    def update_playback_timer(self):
        """定时器更新播放状态"""
        if self.audio_player.current_audio:
            if self.audio_player.is_playing:
                current_time = self.audio_player.get_current_time()
                total_time = self.audio_player.audio_length
                
                # 更新当前时间显示
                self.current_time_label.configure(text=self.format_time(current_time))
                
                # 更新进度条
                if total_time > 0:
                    progress = current_time / total_time
                    self.audio_progress.set(progress)
                
                # 检查是否播放结束
                if self.audio_player.is_audio_ended():
                    self.audio_player.stop()
                    self.play_status_label.configure(text="状态: 播放完成")
                    self.play_btn.configure(text="▶️ 播放")
                else:
                    self.play_status_label.configure(text="状态: 播放中")
            elif self.audio_player.is_paused:
                self.play_status_label.configure(text="状态: 已暂停")
            else:
                self.play_status_label.configure(text="状态: 已停止")
        
        # 每100ms更新一次
        self.after(100, self.update_playback_timer)
        
    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def play_audio(self):
        if self.audio_player.current_audio:
            self.audio_player.play()
            self.play_btn.configure(text="⏸️ 暂停")
            self.update_audio_info()
            
    def pause_audio(self):
        if self.audio_player.is_playing:
            self.audio_player.pause()
            self.play_btn.configure(text="▶️ 播放")
        elif self.audio_player.is_paused:
            self.audio_player.resume()
            self.play_btn.configure(text="⏸️ 暂停")
        self.update_audio_info()
        
    def stop_audio(self):
        self.audio_player.stop()
        self.play_btn.configure(text="▶️ 播放")
        self.audio_progress.set(0)
        self.current_time_label.configure(text="00:00")
        self.update_audio_info()
        
    def on_audio_progress_change(self, value):
        """进度条改变事件"""
        if self.audio_player.current_audio and self.audio_player.audio_length > 0:
            position = value * self.audio_player.audio_length
            self.audio_player.set_position(position)
            
    def on_volume_change(self, value):
        """音量改变事件"""
        pygame.mixer.music.set_volume(value)
        
    def create_audio_info_frame(self, parent):
        """创建音频信息显示框架"""
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_label = ctk.CTkLabel(info_frame, text="🎵 音频信息", font=ctk.CTkFont(size=16, weight="bold"))
        info_label.pack(pady=10)
        
        self.audio_info_label = ctk.CTkLabel(info_frame, text="音频信息: 未加载")
        self.audio_info_label.pack(pady=5)
        
        self.audio_length_label = ctk.CTkLabel(info_frame, text="音频长度: 0s")
        self.audio_length_label.pack(pady=5)
        
        self.sample_rate_label = ctk.CTkLabel(info_frame, text="采样率: 16000Hz")
        self.sample_rate_label.pack(pady=5)
        
    def init_devices(self):
        """初始化设备列表"""
        self.refresh_devices()
        
    def refresh_devices(self):
        """刷新设备列表"""
        try:
            devices = self.tts_generator.get_available_adapters()
            self.device_combo.configure(values=devices)
            if devices:
                self.device_combo.set(devices[0])
        except Exception as e:
            print(f"刷新设备列表失败: {e}")
             
    def browse_model(self):
        """浏览模型文件"""
        from tkinter import filedialog
        path = filedialog.askdirectory(title="选择模型目录")
        if path:
            self.model_path.set(path)
             
    def browse_decoder(self):
        """浏览解码器文件"""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择解码器文件",
            filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")]
        )
        if path:
            self.decoder_path.set(path)
             
    def browse_output(self):
        """浏览输出文件"""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="选择输出文件",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.output_path.set(path)
             
    def on_device_change(self, choice):
        """设备选择改变事件"""
        print(f"选择设备: {choice}")
         
    def on_text_change(self, event):
        """文本变化事件"""
        self.update_language_display()
         
    def update_language_display(self):
        """更新语言显示"""
        text = self.text_input.get("1.0", "end-1c")
        if text.strip():
            import re
            has_zh = re.search(r"[\u4e00-\u9fff]", text) is not None
            has_en = re.search(r"[A-Za-z]", text) is not None
             
            if has_zh and not has_en:
                lang_text = "中文"
            elif has_en and not has_zh:
                lang_text = "英文"
            elif has_zh and has_en:
                lang_text = "中英混合"
            else:
                lang_text = "其他"
        else:
            lang_text = "未输入"
            
        self.lang_label.configure(text=f"检测语言: {lang_text}")
         
    def load_model(self):
        """加载模型"""
        model_path = self.model_path.get()
        if not model_path:
            self.show_error("请选择模型路径")
            return
             
        # 获取选中的设备
        device_name = self.device_combo.get()
        if not device_name:
            self.show_error("请选择设备")
            return
             
        # 查找设备索引
        try:
            if ":" in device_name:
                device_index = int(device_name.split(":")[0])
            else:
                device_index = 0
        except ValueError:
            device_index = 0
         
        # 禁用加载按钮
        self.load_model_btn.configure(state="disabled", text="🔄 加载中...")
         
        # 在新线程中加载模型
        load_thread = threading.Thread(
            target=self.run_model_loading, 
            args=(model_path, device_index)
        )
        load_thread.daemon = True
        load_thread.start()
         
    def run_model_loading(self, model_path: str, device_index: int):
        """运行模型加载（在后台线程中）"""
        try:
            print(f"🔄 开始模型加载流程...")
             
            # 加载模型
            success = self.tts_generator.load_model(
                model_path, 
                device_index, 
                progress_callback=self.update_status
            )
             
            if success:
                print("✅ 模型加载成功，开始加载解码器...")
                 
                # 加载解码器
                decoder_path = self.decoder_path.get()
                if decoder_path and os.path.exists(decoder_path):
                    print(f"🎵 找到解码器路径: {decoder_path}")
                    decoder_success = self.tts_generator.load_decoder(
                        decoder_path,
                        progress_callback=self.update_status
                    )
                    if decoder_success:
                        print("🎉 模型和解码器都加载成功！")
                        self.after(0, self.model_loading_completed)
                    else:
                        print("❌ 解码器加载失败")
                        self.after(0, lambda: self.show_error("解码器加载失败"))
                else:
                    print(f"⚠️ 解码器路径不存在或为空: {decoder_path}")
                    self.after(0, self.model_loading_completed)
            else:
                print("❌ 模型加载失败")
                self.after(0, lambda: self.show_error("模型加载失败"))
                 
        except Exception as e:
            print(f"❌ 模型加载过程中发生错误: {str(e)}")
            self.after(0, lambda: self.show_error(f"模型加载失败: {str(e)}"))
        finally:
            self.after(0, self.model_loading_failed)
             
    def model_loading_completed(self):
        """模型加载完成处理"""
        self.load_model_btn.configure(state="normal", text="✅ 模型已加载")
        self.generate_btn.configure(state="normal")
        self.show_success("模型加载完成！可以开始生成音频")
         
    def model_loading_failed(self):
        """模型加载失败处理"""
        self.load_model_btn.configure(state="normal", text="📥 加载模型")
         
    def start_generation(self):
        """开始生成TTS"""
        text = self.text_input.get("1.0", "end-1c")
         
        if not text.strip():
            self.show_error("请输入要合成的文本")
            return
             
        self.generate_btn.configure(state="disabled", text="🔄 生成中...")
        self.update_progress(0)
        self.update_speed("准备中...")
         
        generation_thread = threading.Thread(
            target=self.run_generation, 
            args=(text,)
        )
        generation_thread.daemon = True
        generation_thread.start()
         
    def run_generation(self, text: str):
        """运行TTS生成（在后台线程中）"""
        try:
            audio_data = self.tts_generator.generate_tts(
                text=text,
                age=self.age_var.get(),
                gender=self.gender_var.get(),
                emotion=self.emotion_var.get(),
                pitch=self.pitch_var.get(),
                speed=self.speed_var.get(),
                progress_callback=self.update_status,
                speed_callback=self.update_speed
            )
             
            if audio_data is not None:
                output_path = self.output_path.get()
                sf.write(output_path, audio_data, 16000)
                self.after(0, lambda: self.generation_completed(output_path))
            else:
                self.after(0, lambda: self.show_error("音频生成失败"))
                 
        except Exception as e:
            self.after(0, lambda: self.show_error(f"生成失败: {str(e)}"))
            self.after(0, self.generation_failed)
             
    def generation_completed(self, output_path: str):
        """生成完成处理"""
        self.generate_btn.configure(state="normal", text="🚀 开始生成")
         
        if self.audio_player.load_audio(output_path):
            self.show_success(f"音频生成成功！已保存到: {output_path}")
            self.update_audio_info()
            # 更新总时间显示
            self.total_time_label.configure(text=self.format_time(self.audio_player.audio_length))
            # 重置进度条
            self.audio_progress.set(0)
        else:
            self.show_error("音频文件加载失败")
             
    def generation_failed(self):
        """生成失败处理"""
        self.generate_btn.configure(state="normal", text="🚀 开始生成")
        self.update_status("生成失败")
        self.update_progress(0)
         
    def update_status(self, status: str, progress: float = None):
        """更新状态显示"""
        self.status_var.set(status)
        if progress is not None:
            self.update_progress(progress)
         
    def update_progress(self, progress: float):
        """更新进度条"""
        self.progress_var.set(progress)
        self.progress_bar.set(progress)
         
    def update_speed(self, speed: str):
        """更新速度显示"""
        self.tokens_per_sec_var.set(speed)
         
    def update_audio_info(self):
        """更新音频信息显示"""
        if self.audio_player.current_audio:
            try:
                audio_data, sample_rate = sf.read(self.audio_player.current_audio)
                self.audio_length_label.configure(text=f"音频长度: {len(audio_data) / sample_rate:.2f}s")
                self.sample_rate_label.configure(text=f"采样率: {sample_rate}Hz")
            except Exception as e:
                print(f"更新音频信息失败: {e}")
                self.audio_length_label.configure(text="音频信息: 加载失败")
                self.sample_rate_label.configure(text="音频信息: 加载失败")
        else:
            self.audio_length_label.configure(text="音频信息: 未加载")
            self.sample_rate_label.configure(text="音频信息: 未加载")
         
    def show_error(self, message: str):
        """显示错误消息"""
        self.status_var.set(f"❌ {message}")
         
    def show_success(self, message: str):
        """显示成功消息"""
        self.status_var.set(f"✅ {message}")

def main():
    app = TTSGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
