from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import argparse
import torch
from model.llm.llm import RWKV7LM
from model.llm.cosy_llm import RWKV7CosyLM,RWKV7CosyConfig
def load_state_dict(ckpt_file, llm):
    llm_state_dict = torch.load(ckpt_file, map_location='cpu')
    #check if self.llm.llm.model.layers[0].attn.x_r exists
    if hasattr(llm.llm.model.layers[0].attn, 'x_r'):
        print(f'x_r exists in self.llm.llm.model.layers[0].attn now convert the checkpoint')
        layer_indices = set()
        for key in llm_state_dict.keys():
            if key.startswith("llm.model.layers."):
                # Extract the layer index from the key
                try:
                    layer_idx = int(key.split(".")[3])  # Extract the number after 'model.layers.'
                    layer_indices.add(layer_idx)
                except ValueError:
                    # Skip keys that don't match the expected format
                    continue

        # Sort the layer indices to process them in order
        sorted_layer_indices = sorted(layer_indices)

        # Migration logic for each layer
        for layer_idx in sorted_layer_indices:
            layer_prefix = f"llm.model.layers.{layer_idx}"
            attn_prefix = f"{layer_prefix}.attn"

            # Check if the layer contains the old 'x_x' parameter
            if f"{attn_prefix}.x_x" in llm_state_dict:
                print(f"Migrating weights for layer {layer_idx} from RWKV7Attention version 1 to version 2...")
                # Extract the x_x parameter
                x_x = llm_state_dict[f"{attn_prefix}.x_x"]
                with torch.no_grad():
                    # Create new parameters for version 2
                    llm_state_dict[f"{attn_prefix}.x_r"] = x_x[0].unsqueeze(0).unsqueeze(0)
                    llm_state_dict[f"{attn_prefix}.x_w"] = x_x[1].unsqueeze(0).unsqueeze(0)
                    llm_state_dict[f"{attn_prefix}.x_k"] = x_x[2].unsqueeze(0).unsqueeze(0)
                    llm_state_dict[f"{attn_prefix}.x_v"] = x_x[3].unsqueeze(0).unsqueeze(0)
                    llm_state_dict[f"{attn_prefix}.x_a"] = x_x[4].unsqueeze(0).unsqueeze(0)
                    llm_state_dict[f"{attn_prefix}.x_g"] = x_x[5].unsqueeze(0).unsqueeze(0)
                del llm_state_dict[f"{attn_prefix}.x_x"]
        llm.load_state_dict(llm_state_dict, strict=True)

    else:
        print(f'x_r does not exist in self.llm.llm.model.layers[0].attn')
        llm.load_state_dict(llm_state_dict, strict=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    # fixed params
    """
    sample_rate: 24000
    llm_input_size: 2048
    llm_output_size: 2048
    spk_embed_dim: 192
    qwen_pretrain_path: ''

    # model params
    # for all class/function included in this repo, we use !<name> or !<new> for intialization, so that user may find all corresponding class/function according to one single yaml.
    # for system/third_party class/function, we do not require this.
    llm: !new:model.llm.llm.RWKV7LM
        llm_input_size: !ref <llm_input_size>
        llm_output_size: !ref <llm_output_size>
        speech_token_size: 6561
        length_normalized_loss: True
        lsm_weight: 0
        vocab_size: 65548
        llm: !ref <qwen_pretrain_path>
        sampling: !name:cosyvoice.utils.common.ras_sampling
            top_p: 0.8
            top_k: 25
            win_size: 10
            tau_r: 0.1
    """
    sample_rate = 24000
    llm_input_size = 2048
    llm_output_size = 2048
    spk_embed_dim = 192
    speech_token_size = 6561
    vocab_size = 65548
    from functools import partial  
    from cosyvoice.utils.common import ras_sampling
    sampling = partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
    rwkv7_lm = RWKV7LM(llm_input_size, llm_output_size, speech_token_size, args.model_path, sampling,length_normalized_loss=True,lsm_weight=0,vocab_size=vocab_size)
    rwkv7_lm_config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
    #convert rwkv7_lm_config to kwargs
    kwargs = {}
    for key, value in rwkv7_lm_config.to_dict().items():
        kwargs[key] = value
    config = RWKV7CosyConfig(**kwargs)
    print(config)
    load_state_dict(args.checkpoint_path, rwkv7_lm)
    print(rwkv7_lm)
    kwargs['vocab_size'] = vocab_size
    kwargs['llm_input_size'] = llm_input_size
    kwargs['llm_output_size'] = llm_output_size
    kwargs['speech_token_size'] = speech_token_size
    kwargs['length_normalized_loss'] = True
    kwargs['lsm_weight'] = 0
    kwargs['mix_ratio'] = [5, 15]
    kwargs['drop_ratio'] = 0.0
    config = RWKV7CosyConfig(
        **kwargs
    )
    config.architectures = ['RWKV7ForSpeech']
    config.auto_map = {
        "AutoConfig": "cosy_llm.RWKV7CosyConfig",
        "AutoModel": "cosy_llm.RWKV7CosyLM",
        "AutoModelForCausalLM": "cosy_llm.RWKV7CosyLM"
    }
    cosy_llm = RWKV7CosyLM(config)
    print(cosy_llm)
    #start to copy weights from rwkv7_lm to cosy_llm
    # 获取源模型的状态字典
    state_dict = rwkv7_lm.state_dict()

    # 创建新的状态字典，移除'llm.'前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('llm.'):
            new_key = key[4:]  # 移除'llm.'前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    new_state_dict['text_embedding.weight'] = state_dict['llm.model.embeddings.weight']
    # 加载调整后的状态字典
    cosy_llm.load_state_dict(new_state_dict, strict=True)
    from utils.utilities import get_tokenizer
    tokenizer = get_tokenizer(args.model_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"save tokenizer to {args.output_path}")
    # 保存转换后的模型
    import shutil
    import os
    cosy_llm.save_pretrained(args.output_path, safe_serialization=False)  # 添加safe_serialization=False来处理共享权重
    shutil.copy(os.path.join(os.path.dirname(__file__), "cosy_llm.py"), args.output_path)
    print(f"copy cozy_llm.py to {args.output_path}")
    # copy hf_rwkv_tokenizer.py from model_dir to output_dir
    shutil.copy(os.path.join(args.model_path, "hf_rwkv_tokenizer.py"), args.output_path)
    print(f"copy hf_rwkv_tokenizer.py to {args.output_path}")
    # copy *txt from model_dir to output_dir
    for file in os.listdir(args.model_path):
        if file.endswith('.txt'):
            shutil.copy(os.path.join(args.model_path, file), os.path.join(args.output_path, file))
            print(f"Copied {file} to {args.output_path}")
    # 重新加载保存的模型进行验证
    print("\n开始验证保存的模型...")
    loaded_cosy_llm = AutoModelForCausalLM.from_pretrained(args.output_path,trust_remote_code=True)
    print("模型重新加载完成")
    
    def map_param_name(name):
        """将rwkv7_lm的参数名映射到cosy_llm的参数名"""
        if name.startswith('llm.'):
            name = name[4:]  # 移除'llm.'前缀
        return name
    
    def compare_models(model1, model2, name1="rwkv7_lm", name2="loaded_cosy_llm"):
        print(f"\n比较 {name1} 和 {name2} 的参数...")
        all_match = True
        total_params = 0
        mismatch_params = 0
        
        # 获取两个模型的状态字典
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        # 创建参数名映射
        mapped_state_dict1 = {map_param_name(k): v for k, v in state_dict1.items()}
        
        # 检查所有参数
        for key2 in state_dict2.keys():
            if key2 not in mapped_state_dict1:
                print(f"参数缺失: {key2} 在 {name1} 中不存在")
                all_match = False
                continue
                
            param1 = mapped_state_dict1[key2]
            param2 = state_dict2[key2]
            total_params += 1
            
            # 检查形状
            if param1.shape != param2.shape:
                print(f"形状不匹配 {key2}: {param1.shape} vs {param2.shape}")
                all_match = False
                mismatch_params += 1
                continue
            
            # 检查值
            if not torch.allclose(param1, param2, rtol=1e-5, atol=1e-5):
                print(f"值不匹配 {key2}")
                print(f"最大差异: {(param1 - param2).abs().max().item()}")
                all_match = False
                mismatch_params += 1
        
        # 检查是否有多余的参数
        mapped_keys = set(mapped_state_dict1.keys())
        model2_keys = set(state_dict2.keys())
        extra_keys = mapped_keys - model2_keys
        if extra_keys:
            print("\n在源模型中存在但目标模型中不存在的参数:")
            for key in extra_keys:
                print(f"多余参数: {key}")
            all_match = False
        
        # 打印比较结果
        print(f"\n比较结果:")
        print(f"总参数数量: {total_params}")
        print(f"不匹配参数数量: {mismatch_params}")
        print(f"参数匹配率: {((total_params - mismatch_params) / total_params * 100):.2f}%")
        print(f"整体匹配状态: {'✓' if all_match else '✗'}")
        
        return all_match
    
    # 比较原始rwkv7_lm和重新加载的cosy_llm
    compare_models(rwkv7_lm, loaded_cosy_llm)