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
    
    speech_tokens = torch.tensor([  [4299, 4299, 6486, 4218, 5835, 5919, 5838, 3732, 4299, 2112, 5407, 4443, 4606, 4499, 539, 1916, 4130, 4777, 5482, 4678, 4993, 4966, 6261, 5650, 4776, 5918, 5032, 5203, 6048, 2331, 4555, 6258, 2247, 5115, 1463, 107, 5649, 4452, 4490, 2366, 5241, 2406, 227, 711, 1917, 5588, 5534, 6261, 300, 2754, 2770, 1703, 2918, 4388, 5853, 3984, 2112, 1029, 2997, 4716, 5006, 632, 5968, 6048, 4365, 5826, 381, 2243, 58, 5649, 4686, 4821, 4382, 1564, 1041, 5322, 3721, 1616, 1601, 2112, 4299, 6078, 5835, 5835, 3651, 1545, 4299, 3435, 4676, 555, 1133, 3637, 3612, 4608, 4850, 238, 3718, 5101, 4517, 3107, 3879, 4367, 724, 3, 2390, 5189, 4778, 5668, 305, 17, 1968, 4443, 6554, 2827, 63, 380, 573, 648, 2766, 5773, 6261, 4686, 4769, 4459, 4248, 6534, 4779, 5108, 1644, 2976, 891, 506, 626, 2906, 2168, 2032, 1701, 5835, 5838, 1788, 4299, 2023, 3481, 3482, 321, 546, 4677, 971, 623, 2906, 5091, 5802, 6533, 2186, 2186, 1457, 2167, 3888, 3648, 1869, 1842, 5319, 5256, 6062, 3713, 6048, 5238, 876, 923, 6401, 6481, 5509, 4349, 3644, 2186, 2186, 1370, 6048, 5968, 3699, 806, 240, 4676, 4488, 2454, 4777, 66, 2216, 326, 758, 1759, 2031, 1951, 5020, 5100, 2753, 3462, 4380, 4471, 4390, 1806, 4677, 968, 5099, 2163, 2594, 5840, 4606, 4920, 971, 1445, 5887, 5319, 857, 5985, 4616, 4844, 1355, 1978, 1946, 6481, 6239, 3881, 3718, 2591, 4966, 1259, 3445, 5019, 5100, 5823, 4345, 2185, 1370, 1563, 512, 278, 2195, 4468, 3666, 4299, 2112, 546, 4923, 2900, 2906, 1450, 224, 2582, 1869, 186, 5189, 4463, 5020, 2752, 62, 2213, 3912, 4443, 6554, 5095, 306, 29, 5940, 6048, 2970, 1535, 555, 4452, 4733, 4480, 1239, 5325, 6000, 5559, 1131, 4920, 4858, 4436, 4462, 5124, 6486, 6159, 4299, 3975, 5835, 6159, 3975, 4299, 4218, 3481, 5020, 4498, 5937, 4200, 6131, 6373, 1774, 3233, 74, 1514, 2031, 3645, 5835, 5838, 2922, 2112, 3462, 4674, 5668, 5101, 6317, 4126, 6073, 2006, 1978, 5020, 5100, 2753, 3471, 4407, 4471, 4378, 5937, 2733, 159, 1856, 5098, 2172, 2621, 3650, 4570, 546, 2337, 2171, 1766, 6048, 802, 3879, 4607, 2900, 1358, 2006, 6401, 6481, 1676, 1531, 2348, 4994, 1259, 4993, 5073, 6069, 6532, 2159, 1427, 1563, 512, 278, 4382, 2937, 3669, 11, 3653, 4479, 4155, 4677, 4598, 5671, 5046, 4345, 2186, 1126, 69, 5020, 2428, 3702, 3395, 5101, 153, 64, 433, 4314, 4830, 4462, 829, 2175, 703, 1463, 815, 3776, 6048, 6291, 5823, 5828, 1615, 5730, 3885, 6074, 1700, 1610, 2112, 3651, 5838, 5838, 5838, 5838, 3975, 4299, 1461, 2349, 5841, 5921, 4597, 2976, 2417, 1463, 26, 1806, 4675, 4461, 1267, 3482, 4777, 1735, 1725, 290, 1403, 3587, 4750, 5100, 2195, 1081, 6537, 2872, 6401, 5995, 951, 2334, 818, 3647, 17, 1763, 1942, 4130, 3077, 1482, 362, 2861, 5772, 6016, 4994, 4561, 6048, 6048, 3150, 5081, 5070, 5985, 6553, 5828, 315, 29, 2972, 6048, 5238, 884, 720, 648, 5044, 5776, 5772, 1250, 2032, 2786, 719, 1457, 2186, 2760, 2673, 2017, 551, 4567, 4432, 4678, 314, 4597, 60, 4713, 1055, 5027, 690, 669, 4808, 4540, 4803, 4675, 4650, 3482, 3481, 314, 35, 3022, 555, 2508, 1697, 2912, 5102, 2087, 2112, 4299, 4215, 5832, 5832, 5835, 3645, 2028, 4299, 4372, 5094, 4606, 2420, 2166, 703, 4133, 656, 3638, 3705, 5338, 4736, 5771, 5976, 6292, 3384, 5826, 1126, 2733, 4128, 3076, 297, 29, 948, 4515, 2344, 4576, 4560, 5223, 5319, 2973, 809, 4444, 5483, 4678, 2753, 2618, 1802, 1741, 4299, 4299, 4299, 4299, 4218, 1987, 1611, 80, 386, 4463, 4560, 4515, 6054, 3129, 4130, 5426, 4512, 4678, 2751, 2834, 188, 1805, 2112, 4299, 5838, 5838, 2112, 2095, 4130, 3887, 3401, 2006, 2014, 4939, 3481, 5019, 960, 2976, 5319, 5337, 3884, 5182, 4754, 4678, 1294, 2807, 914, 2047, 4299, 4299, 4218, 3975, 4299, 2031, 3747, 38, 5840, 4479, 4155, 4686, 4571, 5671, 5688, 1230, 180, 4771, 2243, 1599, 2509, 2753, 969, 5970, 3395, 5090, 144, 2252, 2432, 19, 3666, 524, 2861, 5775, 3852, 4535, 4940, 3096, 6504, 5772, 5802, 3557, 1862, 1861, 6074, 231, 4593, 5832, 3646, 1463, 1564, 1833, 5967, 3708, 1535, 3803, 5329, 1598, 2112, 3645, 3648, 4299, 4299, 2031, 1560, 1477, 572, 5776, 5040, 3798, 2267, 2726, 4068, 6504, 5043, 5803, 1370, 1862, 1861, 3887, 5341, 4920, 2322, 2673, 1403, 2126, 4750, 5019, 4598, 923, 3792, 1688, 41, 4642, 6532, 4127, 218, 8, 4466, 3421, 486, 2511, 632, 2305, 3750, 515, 674, 5775, 6501, 3104, 2726, 4427, 5938, 5979, 869, 2212, 6423, 5487, 3885, 5344, 6261, 5970, 4293, 1073, 3749, 5896, 5319, 3151, 5081, 5098, 6282, 6310, 2186, 2186, 2087, 1944, 5835, 5838, 5838, 1869, 2112, 60, 2324, 533, 4232, 6048, 6292, 4851, 5097, 1125, 4920, 3885, 6074, 4130, 1862, 1843, 3894, 4299, 4299, 2112, 5650, 4533, 4857, 2699, 4675, 4650, 1052, 5935, 4317, 1677, 4607, 5024, 5085, 3702, 4124, 605, 593, 4947, 2769, 5753, 6077, 1703, 1490, 2032, 1943, 4130, 1616, 1407, 660, 4853, 4606, 5020, 3481, 2753, 5581, 5028, 4956, 6481, 3647, 1599, 6049, 4293, 2186, 5095, 3798, 2510, 566, 2086, 2112, 4218, 3975, 5838, 5838, 5835, 4299, 2031, 5136, 5326, 3390, 5344, 4780, 1382, 3224, 6049, 4104, 5096, 5059, 4311, 2807, 2942, 1719, 4406, 5161, 5319, 2322, 3803, 1688, 1598, 1519, 4299, 4299, 1431, 5752, 6157, 5185, 4849, 4912, 4804, 6505, 654, 5830, 5050, 6067, 4664, 2432, 2128, 2127, 518, 2693, 5532, 573, 2886, 2186, 1126, 4191, 6381, 4447, 147, 2973, 3803, 1608, 3870, 3377, 5090, 5098, 1765, 2112, 5649, 4921, 2753, 2222, 8, 1644, 4515, 3081, 4961, 6502, 2148, 2160, 5509, 6157, 5995, 948, 3051, 5582, 5100, 2894, 2003, 1946, 6238, 4780, 246, 162, 2504, 5643, 4914, 4286, 5743, 1923, 2238, 4940, 1294, 546, 5325, 6054, 3806, 6317, 3401, 1763, 2031, 5835, 5838, 5838, 2922, 4299, 4164, 4679, 2508, 4760, 2195, 834, 4512, 3102, 5060, 3422, 1975, 1951, 6158, 5508, 468, 162, 2449, 4914, 4995, 6553, 5826, 570, 2592, 4772, 542, 1271, 1513, 4299, 4191, 6374, 3960, 4447, 5310, 5230, 2189, 3232, 4995, 3546, 5828, 2910, 489, 2349, 4987, 5537, 1676, 1928, 1416, 405, 2530, 2567, 1038, 6374, 6147, 4770, 4191, 6381, 5418, 2344, 1527, 3799, 1612, 3702, 3872, 5090, 5094, 1842, 4678, 2753, 737, 1555, 4434, 894, 2693, 6502, 243, 2592, 4959, 461, 1599, 3141, 704, 5084, 5101, 2087, 2113, 1701, 3648, 5838, 5109, 3651, 2112, 0, 162, 6084, 6158, 5266, 5241, 5238, 2992, 1842, 2265, 5021, 403, 5001, 6000, 6316, 2427, 4920, 2509, 35, 1564, 2247, 3108, 4880, 6502, 162, 5841, 6157, 5834, 166, 2976, 5319, 3051, 1616, 3884, 947, 1517, 2112, 3888, 5832, 3645, 3645, 1701, 2031, 4860, 2511, 1370, 4997, 4591, 1032, 4461, 2555, 4382, 3831, 57, 5101, 2753, 645, 5001, 6028, 3840, 6317, 6074, 5587, 2006, 2112, 5650, 4678, 2744, 8, 1483, 1050, 4593, 2925, 4979, 4960, 2682, 2754, 2017, 1289, 4754, 138, 4380, 2725, 305, 2924, 2192, 1501, 4299, 2112, 1275, 4677, 5344, 5098, 2841, 3557]], device=device, dtype=torch.long)
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
    