import click
import onnxruntime as ort
import numpy as np
import json
@click.command()
@click.option("--decoder_path", type=str, required=False,default='/Volumes/bigdata/models/BiCodecDetokenize.onnx')
@click.option("--input_json", type=str, required=False, default='input_tokens.json')
def main(decoder_path, input_json):
    print(f"🎿Load input_json: {input_json}")
    print(f'🎿Load decoder_path: {decoder_path}')
    with open(input_json, 'r') as f:
        data = json.load(f)
    global_tokens = data['global_tokens']
    semantic_tokens = data['semantic_tokens']
    print(f'🎿global_tokens: {global_tokens}')
    print(f'🎿semantic_tokens: {semantic_tokens}')
    print(f'⛷️Start to load onnx model')
    ort_session = ort.InferenceSession(decoder_path)
    print(f'⛷️Load onnx model success')
    num_inputs = len(ort_session.get_inputs())
    print(f'⛷️num_inputs: {num_inputs}')
    for i in range(num_inputs):
        input_name = ort_session.get_inputs()[i].name
        input_shape = ort_session.get_inputs()[i].shape
        print(f'🎿input_name: {input_name}')
        print(f'🎿input_shape: {input_shape}')
    num_outputs = len(ort_session.get_outputs())
    print(f'⛷️num_outputs: {num_outputs}')
    for i in range(num_outputs):
        output_name = ort_session.get_outputs()[i].name
        output_shape = ort_session.get_outputs()[i].shape
        print(f'🎿output_name: {output_name}')
        print(f'🎿output_shape: {output_shape}')
    global_tokens = np.array(global_tokens, dtype=np.int64).reshape(1,1,-1)
    semantic_tokens = np.array(semantic_tokens, dtype=np.int64).reshape(1,-1)
    outputs = ort_session.run(None, {"global_tokens": global_tokens, "semantic_tokens": semantic_tokens})
    print(f'🎿outputs: {outputs}')
    import soundfile as sf
    wav_reconstructed = outputs[0].reshape(-1)
    sf.write("from_spark_audio_tokens_onnx.wav", wav_reconstructed, 16000)
if __name__ == "__main__":
    main()