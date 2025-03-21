import io
from typing import Optional
import logging
import sys
import argparse

# 配置日志级别
logging.basicConfig(level=logging.WARNING)  # 将日志级别设置为WARNING或更高
# 特别设置可能产生这些消息的库的日志级别
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "starlette", "fastapi"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该更严格
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tts_service import TTS_Service
tts_service = None
def initialize_tts_service(model_path, device_list, threads_per_device):
    global tts_service
    tts_service = TTS_Service(
        model_path=model_path,
        device_list=device_list,
        threads_per_device=threads_per_device
    )
    #需要对tts_service 进行几次请求预热
    if True:
        print('预热')
        prompt_audio_file = 'new.wav'
        text = '这是一个测试'
        prompt_text = '少年强则中国强。'
        try:
            with open(prompt_audio_file, 'rb') as f:
                prompt_audio_data = f.read()
        except FileNotFoundError:
            print(f"预热失败: 找不到文件 {prompt_audio_file}")
            return  # 退出初始化
        
        for i in range(2*1):
            data = tts_service.tts(text, prompt_text, prompt_audio_data, 'wav')
        print('预热完成')
        del prompt_audio_data, text, prompt_text, prompt_audio_file
class TTSResponse(BaseModel):
    audio: bytes
    sample_rate: int = 24000
    
@app.get("/api/speakers")
async def list_speakers():
    """
    列出可用的说话人ID。
    """
    try:
        return list(tts_service.speaker_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取说话人列表失败: {str(e)}")
    
@app.post("/api/rwkv_tts")
async def rwkv_tts(
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_audio: Optional[UploadFile] = File(None),
    audio_format: str = Form("wav"),
    ref_voice: Optional[str] = Form(None)
):
    # 读取上传的音频文件（如果有）
    prompt_audio_bytes = None
    if prompt_audio:
        prompt_audio_bytes = await prompt_audio.read()
    logger.info(f"Processing {text} with prompt_text: {prompt_text}")
    try:
        # 调用TTS服务
        result = tts_service.tts(
            text=text,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio_bytes,
            audio_format=audio_format,
            ref_voice=ref_voice
        )
        
        # 设置响应的内容类型
        if audio_format.lower() == "wav":
            content_type = "audio/wav"
        else:
            content_type = "audio/mpeg"
            
        
        # 创建自定义响应，添加必要的头信息
        headers = {
            "Content-Disposition": f"attachment; filename=result.{audio_format.lower()}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=0"
        }
        # 返回音频数据
        return Response(
            content=result['audio'], 
            media_type=content_type,
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 处理失败: {str(e)}")

@app.post("/api/rwkv_tts_instruct")
async def rwkv_tts_instruct(
    text: str = Form(...),
    instruct: Optional[str] = Form(None),
    prompt_audio: Optional[UploadFile] = File(None),
    audio_format: str = Form("wav"),
    ref_voice: Optional[str] = Form(None)
):
    logger.info(f"Inscripted Processing {text} with instruct: {instruct}, ref_voice: {ref_voice}, audio_format: {audio_format}, prompt_audio: {prompt_audio}")
    # 读取上传的音频文件（如果有）
    prompt_audio_bytes = None
    if prompt_audio:
        prompt_audio_bytes = await prompt_audio.read()
    
    try:
        logger.info(f"Processing {text} with instruct: {instruct}")
        # 处理instruct参数，拼接<|endofprompt|>和text
        processed_text = text
        if instruct:
            processed_text = instruct + "<|endofprompt|>" + text
        logger.info(f"Processed text: {processed_text}")
        # 调用TTS服务，不传递prompt_text
        result = tts_service.tts(
            text=processed_text,
            prompt_text=None,  # 设置为None，不使用prompt_text
            prompt_audio=prompt_audio_bytes,
            audio_format=audio_format,
            ref_voice=ref_voice
        )
        
        # 设置响应的内容类型
        if audio_format.lower() == "wav":
            content_type = "audio/wav"
        else:
            content_type = "audio/mpeg"
            
        # 创建自定义响应，添加必要的头信息
        headers = {
            "Content-Disposition": f"attachment; filename=result.{audio_format.lower()}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=0"
        }
        # 返回音频数据
        return Response(
            content=result['audio'], 
            media_type=content_type,
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 处理失败: {str(e)}")
      
@app.on_event("shutdown")
async def shutdown_event():
    tts_service.shutdown()
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/yueyulin/models/CosyVoice2-0.5B-RWKV-7-1.5B-Instruct-CHENJPKO/", help="模型路径")
    parser.add_argument("--device_list", type=str, default="cuda:0", help="设备列表，用逗号分隔")
    parser.add_argument("--threads_per_device", type=int, default=2, help="每个设备的线程数")
    
    args = parser.parse_args()
    model_path = args.model_path
    device_list = args.device_list.split(",")
    threads_per_device = args.threads_per_device
    print(f"使用参数: model_path={model_path}, device_list={device_list}, threads_per_device={threads_per_device}")
    
    initialize_tts_service(model_path, device_list, threads_per_device)
    uvicorn.run(app, host="0.0.0.0", port=8000)