import click
import glob
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

@click.command()
@click.option("--corpus-path", type=click.Path(exists=True), required=True)
def main(corpus_path):
    """
    Calculate statistics for a corpus.
    """ 
    print(f"Calculating statistics for corpus at {corpus_path}...")

    jsonl_files = glob.glob(os.path.join(corpus_path, "**/*.jsonl"))
    print(f"Found {len(jsonl_files)} jsonl files")
    
    # 基础统计
    semantic_tokens_lengths = []
    text_lengths = []
    
    # 新增统计维度
    gender_pitch_stats = defaultdict(list)  # 按性别统计pitch
    gender_age_stats = defaultdict(list)    # 按性别统计年龄
    gender_age_pitch_stats = defaultdict(list)  # 按性别+年龄统计pitch
    
    tqdm_jsonl_files = tqdm(jsonl_files, desc="Processing jsonl files")
    for jsonl_file in tqdm_jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in tqdm(f, desc="Processing lines"):
                data = json.loads(line)
                len_of_semantic_tokens = len(data["semantic_tokens"])
                len_of_text = len(data["text"])
                semantic_tokens_lengths.append(len_of_semantic_tokens)
                text_lengths.append(len_of_text)
                
                # 新增统计维度
                gender = data.get("gender", "unknown")
                age = data.get("age", "unknown")
                pitch = data.get("pitch", 0.0)
                
                # 按性别统计pitch
                gender_pitch_stats[gender].append(pitch)
                
                # 按性别统计年龄
                gender_age_stats[gender].append(age)
                
                # 按性别+年龄统计pitch
                gender_age_key = f"{gender}_{age}"
                gender_age_pitch_stats[gender_age_key].append(pitch)

    # 基础统计输出
    print("\n=== 基础统计 ===")
    print(f"Average length of semantic tokens: {np.mean(semantic_tokens_lengths)}")
    print(f"Average length of text: {np.mean(text_lengths)}")
    print(f"Max length of semantic tokens: {np.max(semantic_tokens_lengths)}")
    print(f"Max length of text: {np.max(text_lengths)}")
    print(f"Min length of semantic tokens: {np.min(semantic_tokens_lengths)}")
    print(f"Min length of text: {np.min(text_lengths)}")
    print(f"Median length of semantic tokens: {np.median(semantic_tokens_lengths)}")
    print(f"Median length of text: {np.median(text_lengths)}")
    print(f"Standard deviation of semantic tokens: {np.std(semantic_tokens_lengths)}")
    print(f"Standard deviation of text: {np.std(text_lengths)}")
    print(f"All data lengths: {len(semantic_tokens_lengths)}")
    
    # 1. 按性别统计pitch分布
    print("\n=== 按性别统计pitch分布 ===")
    for gender, pitches in gender_pitch_stats.items():
        if len(pitches) > 0:
            print(f"\n性别: {gender}")
            print(f"  样本数量: {len(pitches)}")
            print(f"  平均pitch: {np.mean(pitches):.2f}")
            print(f"  中位数pitch: {np.median(pitches):.2f}")
            print(f"  标准差: {np.std(pitches):.2f}")
            print(f"  最小值: {np.min(pitches):.2f}")
            print(f"  最大值: {np.max(pitches):.2f}")
            print(f"  25%分位数: {np.percentile(pitches, 25):.2f}")
            print(f"  75%分位数: {np.percentile(pitches, 75):.2f}")
    
    # 2. 按性别统计年龄分布
    print("\n=== 按性别统计年龄分布 ===")
    for gender, ages in gender_age_stats.items():
        if len(ages) > 0:
            age_counter = Counter(ages)
            print(f"\n性别: {gender}")
            print(f"  样本数量: {len(ages)}")
            print(f"  年龄分布:")
            for age, count in age_counter.most_common():
                percentage = (count / len(ages)) * 100
                print(f"    {age}: {count} ({percentage:.1f}%)")
    
    # 3. 按性别+年龄统计pitch分布
    print("\n=== 按性别+年龄统计pitch分布 ===")
    for gender_age_key, pitches in gender_age_pitch_stats.items():
        if len(pitches) > 0:
            gender, age = gender_age_key.split("_", 1)
            print(f"\n性别: {gender}, 年龄: {age}")
            print(f"  样本数量: {len(pitches)}")
            print(f"  平均pitch: {np.mean(pitches):.2f}")
            print(f"  中位数pitch: {np.median(pitches):.2f}")
            print(f"  标准差: {np.std(pitches):.2f}")
            print(f"  最小值: {np.min(pitches):.2f}")
            print(f"  最大值: {np.max(pitches):.2f}")
            print(f"  25%分位数: {np.percentile(pitches, 25):.2f}")
            print(f"  75%分位数: {np.percentile(pitches, 75):.2f}")

if __name__ == "__main__":
    main()