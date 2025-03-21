# RWKV TTS API 服务使用说明

## 概述
 RWKV TTS 是一个基于 RWKV 模型的文本转语音服务，使用 CosyVoice2 作为底层引擎。该服务支持通过提示音频和文本进行语音克隆，可以生成高质量的语音合成结果。

## API 端点
### 文本转语音
#### 端点：POST /api/rwkv_tts
将文本转换为语音，支持使用提示音频和文本引导生成的声音特征。

#### 请求参数
| 参数名 | 类型 | 必填 |描述 |
| --- | --- | --- | --- |
| text | string | 是 | 要转换为语音的文本内容 |
| prompt_text | string | 否 | 提示文本，帮助模型理解提示音频对应的文字内容 |
| prompt_audio | file | 是 | 提示音频文件，模型将尝试模仿该音频的声音特征 |
| audio_format | string | 否 | 输出音频格式，支持 "wav" 或 "mp3"，默认为 "wav" |

#### 响应
成功时返回生成的音频文件，HTTP 状态码为 200。

#### 响应头：
```
Content-Type: audio/wav 或 audio/mpeg (根据请求的 audio_format)
Content-Disposition: attachment; filename=result.wav 或 result.mp3
Accept-Ranges: bytes
Cache-Control: public, max-age=0
```

#### 错误响应： 如果处理过程中发生错误，将返回 HTTP 状态码 500 和错误详情。

### 使用示例
#### cURL
```bash
# 基本文本转语音请求，使用WAV格式
curl -X POST http://localhost:8000/api/rwkv_tts \
  -F "text=这是一个语音合成示例" \
  -F "prompt_text=这是提示文本" \
  -F "prompt_audio=@/path/to/reference_audio.wav" \
  -F "audio_format=wav" \
  --output generated_speech.wav

# 使用MP3格式
curl -X POST http://localhost:8000/api/rwkv_tts \
  -F "text=这是另一个语音合成示例" \
  -F "prompt_audio=@/path/to/reference_audio.wav" \
  -F "audio_format=mp3" \
  --output generated_speech.mp3
```

### 注意事项
1. 提示音频是必需的，服务将使用它来克隆语音特征
2. 提示文本是可选的，但提供它可以帮助模型更好地理解提示音频。没有的话就跟随语音语调，适合跨语言使用。
3. 长文本可能需要更长的处理时间
4. 服务默认有600秒(10分钟)的处理超时时间


#### 端点：/api/rwkv_tts_instruct

根据指示生成语音

#### 请求参数
| 参数名 | 类型 | 必填 |描述 |
| --- | --- | --- | --- |
| text | string | 是 | 要转换为语音的文本内容 |
| audio_format | string | 否 | 输出音频格式，支持 "wav" 或 "mp3"，默认为 "wav" |
|instruct|string|是|输出的指示|
|prompt_audio|file|否|提示音频文件，模型将尝试模仿该音频的声音特征|
|ref_voice｜string|否|内置的参考音色，模型将尝试模仿该音频的声音特征,ref_voice 和 prompt_audio 两个必须有一个不为空，prompt_audio 不为空的时候，ref_voice 不生效|

 #### 响应
成功时返回生成的音频文件，HTTP 状态码为 200。

#### 指示说明：

| 指示 | 描述 | 例子 |
| --- | --- | --- |
|情感|高兴(Happy), 悲伤(Sad), 惊讶(Surprised), 愤怒(Angry), 恐惧(Fearful), 厌恶(Disgusted), 冷静(Calm), 严肃(Serious)|instruct: 你能用高兴的情感说吗？<br> text:今天真是太开心了，马上要放假了！|
|方言|粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话|instruct: 请问你能模仿粤语的口音吗？<br> text:我要去买菜|
|语速|快速(Fast), 非常快速(Very Fast), 慢速(Slow), 非常慢速(Very Slow)|instruct: 请用非常快速的语速说一下<br> text:我要去买菜|
|角色扮演|神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely), 机器人(Robot), 小猪佩奇(Peppa), etc.| instruct: 请用小猪佩奇的声音说一下<br> text:我要去买菜|
|语气词|'[breath]', '[noise]','[laughter]', '[cough]', '[clucking]', '[accent]','[quick_breath]',"[hissing]", "[sigh]", "[vocalized-noise]","[lipsmack]", "[mn]"|instruct: 请用小猪配齐的声音说话<br> text:我要去买菜[laughter]|
|说日语|日本語で話してください。|instruct:日本語で話してください。text:いろんな食べ物があって、いいよね。 |
|说韩语|한국어로 말씀해주세요.|instruct:한국어로 말씀해주세요。text:오늘은 날씨가 좋아요. |

#### cURL 例子：
```bash
curl -X 'POST' \
  'https://fastapi.rwkvos.com/api/rwkv_tts_instruct' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'text=[noise]马上要放假了！[laughter]' \
  -F 'instruct=请用小猪佩奇的声音说一下' \
  -F 'prompt_audio=@mine.wav;type=audio/wav' \
  -F 'audio_format=wav'
  ``` 

#### 端点：/api/speakers
获取所有内置的音色
```bash
curl -X 'GET' \
  'https://fastapi.rwkvos.com/api/speakers' \
  -H 'accept: application/json'
```

返回
```json
[
  "Qingzu_Korean",
  "Local Diner_English",
  "Jessamy_Korean",
  "Tailor_Korean",
  "Master of the Disciples of Sanctus Medicus_English",
  "Ruoyue_Japanese",
  "Argenti_Korean",
  "Yanming_English",
  "Timid Researcher_Korean",
]
```