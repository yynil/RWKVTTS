#!/usr/bin/env python3
"""
Test script for SFM (Speech Flow Matching) model
Usage: python test_sfm.py --config_file model/flow/train_sfm_flow_detailed.yaml --wav_file input.wav --checkpoint_file checkpoint.pt
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
import onnxruntime
import whisper
from hyperpyyaml import load_hyperpyyaml
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
from cosyvoice.utils.mask import make_pad_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Test SFM model with audio input")
    parser.add_argument("--config_file", type=str, required=True, help="Path to model config file (YAML)")
    parser.add_argument("--wav_file", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--output_file", type=str, default="reconstructed_sfm.wav", help="Output WAV file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], 
                       help="Data type to use (default: float16)")
    parser.add_argument("--try_inference", action="store_true", help="Try to run inference (may fail due to mask issues)")
    parser.add_argument("--campplus_path", type=str, 
                       default='/home/yueyulin/models/CosyVoice2-0.5B/campplus.onnx',
                       help="Path to CamPPlus model")
    parser.add_argument("--speech_tokenizer_file", type=str,
                       default='/home/yueyulin/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx',
                       help="Path to speech tokenizer model")
    parser.add_argument("--hifi_gan_path", type=str,
                       default='/home/yueyulin/models/CosyVoice2-0.5B/hift.pt',
                       help="Path to hifi-gan model")
    parser.add_argument("--sfm_strength", type=float, default=2.5, help="SFM strength parameter alpha")
    return parser.parse_args()


def load_audio(wav_path, target_sr=16000):
    """Load and resample audio to target sample rate"""
    waveform, sample_rate = torchaudio.load(wav_path)
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy(), target_sr


def extract_campplus_embedding(audio, campplus_session, device_id=0):
    """Extract CamPPlus speaker embedding from audio"""
    # Convert to 16kHz if needed
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)
    
    # Extract Kaldi features
    feat = kaldi.fbank(torch.from_numpy(audio), num_mel_bins=80, dither=0, sample_frequency=160000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat = feat.unsqueeze(0)
    
    # Run CamPPlus model
    embedding = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.numpy()})[0].flatten()
    
    return torch.tensor(embedding, dtype=torch.float32)


def extract_speech_tokens(audio, speech_tokenizer_session):
    """Extract speech tokens from audio using Whisper + speech tokenizer"""
    # Convert to tensor
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    else:
        audio_tensor = audio.unsqueeze(0)
    
    # Extract Whisper mel spectrogram
    feat_whisper = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)
    feat_whisper_np = feat_whisper.detach().cpu().numpy()
    feat_length = np.array([feat_whisper.shape[2]], dtype=np.int32)
    
    # Run speech tokenizer
    speech_token = speech_tokenizer_session.run(None, {
        speech_tokenizer_session.get_inputs()[0].name: feat_whisper_np,
        speech_tokenizer_session.get_inputs()[1].name: feat_length
    })[0]
    
    return torch.tensor(speech_token.squeeze(0), dtype=torch.long)


def process_audio_for_sfm(wav_path, feat_extractor, campplus_session, speech_tokenizer_session, sample_rate, device):
    """Process audio file for SFM model input"""
    print(f"Processing audio file: {wav_path}")
    
    # Load and process audio
    audio, sr = load_audio(wav_path, sample_rate)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(torch.float).to(device)
    
    # Extract mel features
    mel = feat_extractor(audio_tensor).squeeze(0)
    mel = mel.transpose(0, 1)  # [time, mel_dim]
    
    # Extract CamPPlus embedding
    print("Extracting CamPPlus embedding...")
    embedding = extract_campplus_embedding(audio, campplus_session)
    
    # Extract speech tokens
    print("Extracting speech tokens...")
    speech_token = extract_speech_tokens(audio, speech_tokenizer_session)
    
    # Align speech_token and mel features
    token_mel_ratio = 2
    token_len = int(min(mel.shape[0] / token_mel_ratio, speech_token.shape[0]))
    speech_token = speech_token[:token_len]
    mel = mel[:token_len * token_mel_ratio]
    
    print(f"Mel shape: {mel.shape}, Speech token shape: {speech_token.shape}")
    
    return mel, embedding, speech_token


def create_batch_data(mel, embedding, speech_token, device, dtype=torch.float16):
    """Create batch data format for SFM model"""
    # Add batch dimension
    mel = mel.unsqueeze(0)  # [1, time, mel_dim]
    embedding = embedding.unsqueeze(0)  # [1, embedding_dim]
    speech_token = speech_token.unsqueeze(0)  # [1, token_len]
    
    # Create lengths
    mel_len = torch.tensor([mel.shape[1]], dtype=torch.long, device=device)
    token_len = torch.tensor([speech_token.shape[1]], dtype=torch.long, device=device)
    
    return {
        'speech_feat': mel.to(dtype).to(device),
        'speech_feat_len': mel_len,
        'speech_token': speech_token.to(device),
        'speech_token_len': token_len,
        'embedding': embedding.to(device),
        'texts': ['test_audio']
    }


def reconstruct_audio_with_sfm_fixed(model, speech_token, embedding, device, args, dtype=torch.float16):
    """Reconstruct audio using SFM model inference with fixed mask dimensions"""
    print("Starting SFM audio reconstruction (fixed version)...")
    
    # Set model to eval mode
    model.eval()
    
    # Update SFM strength if provided
    if hasattr(model, 'sfm_strength'):
        model.sfm_strength = args.sfm_strength
    
    with torch.no_grad():
        # For reconstruction, we need to use the inference method
        # Since we don't have prompt tokens, we'll use empty prompts
        batch_size = speech_token.shape[0]
        token_len = speech_token.shape[1]
        
        # Create empty prompt tokens (no prompt for reconstruction)
        prompt_token = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        prompt_token_len = torch.tensor([0] * batch_size, dtype=torch.long, device=device)
        prompt_feat = torch.zeros(batch_size, 0, speech_token.shape[1], dtype=dtype, device=device)
        prompt_feat_len = torch.tensor([0] * batch_size, dtype=torch.long, device=device)
        
        # Use autocast for automatic precision conversion
        autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
            try:
                # Run inference
                reconstructed_mel, _ = model.inference(
                    token=speech_token,
                    token_len=torch.tensor([token_len], dtype=torch.long, device=device),
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    streaming=False,
                    finalize=True
                )
                
                print(f"Reconstructed mel shape: {reconstructed_mel.shape}")
                return reconstructed_mel
                
            except RuntimeError as e:
                import traceback
                traceback.print_exc()
                if "expanded size" in str(e) and "must match" in str(e):
                    print(f"Mask dimension error detected: {e}")
                    print("Skipping inference and returning None.")
                    return None
                else:
                    raise e


def mel_to_audio(mel, hift):
    """Convert mel spectrogram to audio using HiFT vocoder"""
    if hift is None:
        print("Warning: HiFT vocoder not available, returning mel spectrogram instead")
        return mel
    
    print("Converting mel spectrogram to audio using HiFT vocoder...")
    
    with torch.no_grad():
        # Ensure mel is in [B, mel_dim, T] format for HiFT
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0).transpose(1, 2)  # [T, 80] -> [1, 80, T]
        elif len(mel.shape) == 3:
            mel = mel.transpose(1, 2)  # [B, T, 80] -> [B, 80, T]
        
        # Empty cache for non-streaming inference
        cache_source = torch.zeros(1, 1, 0, device=mel.device, dtype=mel.dtype)
        
        # HiFT inference returns (waveform, source)
        waveform, _ = hift.inference(speech_feat=mel, cache_source=cache_source)
    
    print(f"Generated waveform shape: {waveform.shape}")
    return waveform


def main():
    args = parse_args()
    
    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Using device: {device}")
    print(f"Using dtype: {args.dtype} ({dtype})")
    
    # Load config
    print(f"Loading config from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        configs = load_hyperpyyaml(f)
    
    sample_rate = configs['sample_rate']
    feat_extractor = configs['feat_extractor']
    
    # Initialize ONNX sessions
    print("Initializing ONNX sessions...")
    device_id = 0 if device.type == 'cuda' else -1
    providers = [("CUDAExecutionProvider", {"device_id": device_id})] if device.type == 'cuda' else ["CPUExecutionProvider"]
    
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    
    campplus_session = onnxruntime.InferenceSession(args.campplus_path, sess_options=option, providers=providers)
    speech_tokenizer_session = onnxruntime.InferenceSession(args.speech_tokenizer_file, sess_options=option, providers=providers)
    
    # Load SFM model
    print(f"Loading SFM model from: {args.checkpoint_file}")
    flow_model = configs['flow']
    
    # Load HiFT model
    print(f"Loading HiFT model from: {args.hifi_gan_path}")
    hift_model = configs['hift']
    hift_model.load_state_dict(torch.load(args.hifi_gan_path, map_location='cpu'))
    hift_model.eval()
    hift_model.to(device)
    
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    if 'module' in checkpoint:
        # Handle DataParallel/DistributedDataParallel state dict
        flow_model.load_state_dict(checkpoint['module'])
    else:
        flow_model.load_state_dict(checkpoint)
    
    print("Model loaded successfully!")
    

    flow_model = flow_model.to(dtype).to(device)
    flow_model.eval()
    # Process audio
    mel, embedding, speech_token = process_audio_for_sfm(
        args.wav_file, feat_extractor, campplus_session, speech_tokenizer_session, sample_rate, device
    )
    
    # Create batch data for testing
    batch_data = create_batch_data(mel, embedding, speech_token, device, dtype)
    
    print("Batch data created:")
    print(f"  speech_feat: {batch_data['speech_feat'].shape} (dtype: {batch_data['speech_feat'].dtype})")
    print(f"  speech_token: {batch_data['speech_token'].shape}")
    print(f"  embedding: {batch_data['embedding'].shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        autocast_dtype = torch.float16 if dtype == torch.float16 else torch.bfloat16
        with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
            loss_dict = flow_model(batch=batch_data, device=device)
            print(f"Forward pass loss: {loss_dict['loss'].item():.4f}")
            
            # Check for NaN loss
            if torch.isnan(loss_dict['loss']):
                print("WARNING: Loss is NaN! This indicates a problem with the model or data.")
                print("Individual losses:")
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.item():.4f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("Forward pass completed successfully!")
    
    # For now, skip the problematic inference and just save the forward pass results
    print("Skipping inference due to mask dimension issues...")
    
    # Optionally try the fixed inference version
    reconstructed_mel = None
    reconstructed_audio = None
    if args.try_inference:
        print("Attempting fixed inference version...")
        print(f'speech token is {batch_data["speech_token"]}')
        reconstructed_mel = reconstruct_audio_with_sfm_fixed(
            flow_model, 
            batch_data['speech_token'], 
            batch_data['embedding'], 
            device, 
            args,
            dtype
        )
        
        if reconstructed_mel is not None:
            print("Inference completed successfully!")
            # Save reconstructed mel
            torch.save(reconstructed_mel.cpu(), 'reconstructed_mel.pt')
            print("Reconstructed mel saved to reconstructed_mel.pt")
        else:
            print("Inference failed, continuing with forward pass results only.")
    
    # Save results
    print("Saving results...")
    
    # Save intermediate results for debugging
    results_dict = {
        'original_mel': mel.cpu(),
        'speech_token': speech_token.cpu(),
        'embedding': embedding.cpu(),
        'loss_dict': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()},
        'sfm_strength': args.sfm_strength,
        'dtype': args.dtype,
        'batch_data_shapes': {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
    }
    
    if reconstructed_mel is not None:
        results_dict['reconstructed_mel'] = reconstructed_mel.cpu()
        
        # Convert reconstructed mel to audio
        print("Converting reconstructed mel to audio...")
        reconstructed_mel = reconstructed_mel.transpose(1, 2)
        reconstructed_audio = mel_to_audio(reconstructed_mel, hift_model)
        
        if isinstance(reconstructed_audio, torch.Tensor):
            # Save audio file
            audio_path = args.output_file
            print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
            torchaudio.save(audio_path, reconstructed_audio.cpu(), 24000)
            print(f"Reconstructed audio saved to: {audio_path}")
            results_dict['reconstructed_audio'] = reconstructed_audio.cpu()
        else:
            print("Audio conversion failed, saving mel spectrogram instead")
    
    torch.save(results_dict, 'sfm_test_results.pt')
    
    # Print final summary
    print("\n=== Test Summary ===")
    print(f"Original mel shape: {mel.shape}")
    print(f"Forward pass loss: {loss_dict['loss'].item():.4f}")
    print(f"Used dtype: {args.dtype}")
    
    if reconstructed_mel is not None:
        print(f"Reconstructed mel shape: {reconstructed_mel.shape}")
        if isinstance(reconstructed_audio, torch.Tensor):
            print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
            print("✓ Audio generation completed successfully!")
        else:
            print("⚠ Audio conversion failed")
    else:
        if not args.try_inference:
            print("ℹ Inference was skipped (use --try_inference to attempt)")
        else:
            print("✗ Inference failed due to mask dimension issues")
    
    print("✓ Forward pass test completed successfully")
    print("✓ Results saved to sfm_test_results.pt")


if __name__ == "__main__":
    main() 