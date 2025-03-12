import io
from typing import Optional
import logging
import sys

# 配置日志级别
logging.basicConfig(level=logging.WARNING)  # 将日志级别设置为WARNING或更高
# 特别设置可能产生这些消息的库的日志级别
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "starlette", "fastapi"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
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

# 创建TTS服务
tts_service = TTS_Service(
    model_path="/home/yueyulin/models/CosyVoice2-0.5B_RWKV_1.5B/",
    device_list=["cuda:0"],  # 可以是 ["cuda:0", "cuda:1"] 等
    threads_per_device=2
)
#需要对tts_service 进行几次请求预热
if True:
    print('预热')
    prompt_audio_file = 'new.wav'
    text = '这是一个测试'
    prompt_text = '少年强则中国强。'
    prompt_audio_data = open(prompt_audio_file, 'rb').read()
    for i in range(2*1):
        data = tts_service.tts(text, prompt_text, prompt_audio_data, 'wav')
    print('预热完成')
    del prompt_audio_data, text, prompt_text, prompt_audio_file
class TTSResponse(BaseModel):
    audio: bytes
    sample_rate: int = 24000

@app.post("/api/rwkv_tts")
async def rwkv_tts(
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_audio: Optional[UploadFile] = File(None),
    audio_format: str = Form("wav")
):
    # 读取上传的音频文件（如果有）
    prompt_audio_bytes = None
    if prompt_audio:
        prompt_audio_bytes = await prompt_audio.read()
    
    try:
        # 调用TTS服务
        result = tts_service.tts(
            text=text,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio_bytes,
            audio_format=audio_format
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
    uvicorn.run(app, host="0.0.0.0", port=8000)