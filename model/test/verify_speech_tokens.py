from pyexpat import model
import time


def do_tts(tts_text,prompt_texts,cosyvoice,prefix):
    import logging
    for i, (prompt_audio_file, prompt_text) in enumerate(zip(prompt_audios, prompt_texts)):
        logging.info(f'Processing {prompt_text}')
        prompt_speech_16k = load_wav(prompt_audio_file, 16000)
        with torch.no_grad():
            for j, k in enumerate(cosyvoice.inference_zero_shot(tts_text,prompt_text, prompt_speech_16k, stream=False,speed=1)):
                torch.cuda.manual_seed_all(time.time())
                torchaudio.save('{}_{}_{}.wav'.format(prefix,i, j), k['tts_speech'], cosyvoice.sample_rate)
        logging.info(f'Finished processing {prompt_text}')
if __name__ == '__main__':
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import sys
    # model_path = '/home/yueyulin/models/CosyVoice2-0.5B_RWKV_0.19B/'
    # device = 'cuda:0'
    print(sys.argv)
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
    cosyvoice = CosyVoice2(model_path,device=device,fp16=False,load_jit=False)
    
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio
    #{"text": "请问你能模仿上海话的口音吗？<|endofprompt|>[laughter]太史，是中国古代官名。", "tts_speech_tokens": [3295, 2031, 1734, 1950, 2031, 1707, 1950, 1950, 1950, 1950, 2031, 1977, 1974, 3888, 5832, 2916, 2225, 37, 2916, 1728, 1734, 2031, 1950, 1950, 2112, 2139, 54, 5832, 4142, 6320, 5834, 2276, 4431, 4753, 4429, 4671, 3720, 1534, 1538, 4451, 4519, 1517, 2005, 1732, 1488, 1488, 1704, 1707, 2031, 1950, 2031, 1950, 4137, 1788, 1606, 5238, 4915, 5644, 4671, 1534, 806, 2264, 2993, 3074, 2992, 1356, 5565, 2666, 5074, 5776, 1321, 699, 2906, 5081, 5090, 1410, 651, 2186, 3641, 153, 57, 2303, 2276, 2384, 1206, 654, 2908, 2648, 5189, 2386, 6258, 1882, 377, 539, 620, 3266, 5935, 1966, 1959, 1950, 2031, 2031, 2031, 1707, 1950, 2031, 2031, 2031, 2031, 1950], "prompt_text": "", "llm_prompt_speech_token": []}
    prompt_audios = [
        '/home/yueyulin/github/RWKVTTS/zero_shot_prompt.wav'
    ]
    
    prompt_texts = [
            '请用凶猛的语气说。',
            # '以机器人的角色和我交流。',
            # '请用非常快速的语速说。',
            # '请用愤怒的情感说一下。'
        ]
    prompt_audio_file = prompt_audios[0]
    instruct_text = prompt_texts[0]
    prompt_speech_16k = load_wav(prompt_audio_file, 16000)
    tts_text = "聯合國會員國是聯合國大會的正式成員，在聯合國大會中擁有平等的代表權。截至2021年，聯合國一共有193個會員國。"
    model_input = cosyvoice.frontend.frontend_instruct2(tts_text, instruct_text, prompt_speech_16k, cosyvoice.sample_rate)
    print(model_input)
    # do_tts('By unifying streaming and non-streaming synthesis within a single framework, CosyVoice 2 achieves human parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. ',prompt_texts,cosyvoice,"instructed_en")
    
    # do_tts('一个教授逻辑学的教授，[laughter]有三个学生，而且三个学生均非常聪明！[breath]一天教授给他们出了一个题，[breath]教授在每个人<strong>脑门上</strong>贴了一张纸条并告诉他们，每个人的纸条上都写了一个正整数，且某两个数的和等于第三个！',prompt_texts,cosyvoice,"instructed_original_cn")
    llm_embedding = model_input['llm_embedding']
    flow_embedding = model_input['flow_embedding']
    
    speech_tokens = torch.tensor([[1571, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 1725, 3831, 46, 2867, 3617, 4319, 4313, 6209, 5020, 2330, 4565, 5589, 4872, 2870, 710, 5087, 303, 4754, 2906, 2777, 1644, 387, 1208, 2348, 1356, 5405, 4408, 4731, 1295, 971, 4860, 4876, 5054, 1434, 1686, 647, 35, 2378, 2760, 4956, 6077, 948, 73, 3482, 3455, 5642, 5615, 5046, 4749, 5320, 4677, 4777, 5917, 6077, 992, 1953, 3888, 3645, 1785, 2112, 1869, 2814, 5808, 4287, 5823, 4594, 4862, 702, 705, 1952, 1220, 5057, 690, 2838, 4962, 5050, 710, 1456, 1454, 56, 776, 6175, 5409, 6455, 4986, 4528, 6378, 5405, 4461, 510, 1213, 60, 2697, 2427, 2160, 5832, 5841, 5837, 2924, 2411, 4616, 6210, 6453, 2702, 3404, 488, 506, 2031, 3888, 2112, 3975, 1942, 4130, 5344, 5166, 6536, 6372, 56, 46, 3747, 4751, 5047, 6261, 114, 2598, 5352, 5119, 4463, 4543, 4536, 4735, 6534, 2547, 6157, 3890, 1703, 1712, 2028, 1701, 3972, 5919, 5916, 5835, 3645, 3969, 4215, 4299, 4218, 2049, 6018, 262, 5054, 5048, 6506, 4316, 5936, 5749, 5668, 2582, 2303, 2222, 764, 3975, 6486, 4299, 2112, 4071, 3020, 4993, 5696, 6344, 5695, 4966, 5020, 2330, 2222, 35, 1573, 3648, 3645, 2112, 3567, 5028, 2906, 2414, 3462, 5162, 4488, 2563, 867, 2928, 3655, 3890, 2296, 303, 3321, 3576, 5097, 225, 42, 2834, 1133, 5908, 6454, 6455, 3951, 5182, 4603, 2876, 4319, 3616, 5028, 5047, 2897, 2894, 2584, 300, 2567, 710, 1403, 2129, 2067, 5644, 2922, 5848, 5111, 4567, 6372, 2466, 4430, 1560, 1480, 5021, 6311, 2185, 725, 56, 779, 6255, 5401, 6455, 4743, 2262, 5648, 4435, 4704, 539, 566, 1680, 2220, 2760, 4993, 5102, 314, 1086, 3651, 5838, 3651, 1869, 1815, 5400, 4454, 2924, 4382, 73, 5652, 2243, 2233, 5851, 6016, 1476, 5087, 398, 476, 970, 3462, 2250, 2831, 5043, 6501, 2538, 5832, 5191, 4382, 2309, 6534, 2790, 2678, 3422, 101, 2157, 67, 5021, 2753, 1286, 2140, 3732, 5838, 3651, 1869, 5859, 5482, 6373, 6130, 6372, 3961, 6391, 6453, 6456, 2258, 8, 2924, 4598, 6378, 5412, 4778, 4463, 1191, 2160, 224, 4841, 3060, 6534, 2549, 2209, 6261, 6378, 6381, 4599, 5185, 1978, 1946, 1541, 1652, 1661, 2983, 6536, 6455, 3231, 5993, 5668, 3481, 5669, 5020, 5021, 566, 2086, 2112, 4299, 4299, 4299, 4218, 4299, 6486, 6486, 6486, 4299, 4299, 6486, 4299, 4299, 4299, 4218, 1806, 3828, 2711, 2879, 4319, 3989, 4993, 4940, 2330, 2300, 147, 6377, 2919, 5119, 5111, 4540, 5157, 6372, 4447, 1560, 751, 4607, 5827, 2185, 725, 56, 2936, 5445, 5482, 6454, 4743, 4527, 5648, 4516, 2517, 566, 1599, 2193, 2760, 5020, 2343, 6379, 2503, 2702, 2870, 5066, 5071, 1302, 6318, 1458, 2112, 6378, 5400, 5841, 1220, 5056, 573, 5097, 5086, 4535, 3481, 3482, 5749, 957, 2166, 2300, 251, 251, 737, 812, 2912, 6559, 5096, 2381, 4571, 2582, 2006, 2113, 1294, 5668, 4210, 3481, 35, 5348, 5044, 2778, 5757, 2860, 4946, 3404, 1946, 1541, 1622, 1679, 1679, 5324, 2415, 1761, 4567, 4625, 2924, 737, 1466, 1464, 1701, 1302, 4944, 673, 713, 969, 5650, 2337, 4769, 3653, 3647, 6320, 4973, 3643, 5827, 4595, 4463, 2195, 8, 764, 1789, 4299, 4299, 4299, 4299, 4299, 1788, 5943, 5325, 5326, 3840, 1700, 1616, 2125, 4317, 3617, 5804, 4825, 5100, 5019, 4760, 2303, 2222, 2264, 6454, 6454, 6381, 3962, 748, 3831, 47, 2867, 3590, 4319, 3857, 5020, 5021, 2303, 2327, 3641, 3643, 713, 68, 2936, 5448, 6373, 6455, 4743, 4527, 2004, 5405, 4461, 582, 566, 960, 2220, 2697, 240, 786, 5021, 2267, 1851, 5649, 3303, 4289, 6556, 5344, 4605, 4779, 2678, 515, 5987, 2189, 5840, 2195, 5102, 6316, 6313, 3884, 4598, 4382, 737, 1493, 4299, 4299, 4299, 4299, 4299, 4299, 4299, 4299]], device=device, dtype=torch.long)
    this_uuid = 'xxx'
    model = cosyvoice.model
    model.tts_speech_token_dict[this_uuid], model.llm_end_dict[this_uuid] = [], False
    model.hift_cache_dict[this_uuid] = None
    this_tts_speech = model.token2wav(token=speech_tokens,
                                             prompt_token=torch.zeros(1, 0, dtype=torch.int32,device=device),
                                             prompt_feat=torch.zeros(1, 0, 80,device=device),
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=1.0).cpu()
    torchaudio.save('{}_{}.wav'.format('from_speech_tokens', 0), this_tts_speech, cosyvoice.sample_rate)
    