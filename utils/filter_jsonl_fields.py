#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path

# 程序需要的字段
REQUIRED_FIELDS = {
    'text',
    'global_tokens', 
    'semantic_tokens',
    'age',
    'gender', 
    'emotion',
    'pitch',
    'speed'
}

def filter_jsonl_file(input_file, output_file):
    """过滤jsonl文件，只保留需要的字段"""
    print(f"处理文件: {input_file}")
    
    # 确保output_file是Path对象
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 只保留需要的字段
                    filtered_data = {}
                    for field in REQUIRED_FIELDS:
                        if field in data:
                            filtered_data[field] = data[field]
                        else:
                            print(f"警告: 文件 {input_file} 第 {line_num} 行缺少字段 '{field}'")
                    
                    # 写入过滤后的数据
                    f_out.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"错误: 文件 {input_file} 第 {line_num} 行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    print(f"错误: 处理文件 {input_file} 第 {line_num} 行时出错: {e}")
                    continue
    
    print(f"完成: {input_file} -> {output_path}")

def process_directory(input_dir, output_dir):
    """处理目录下的所有jsonl文件，保持目录结构"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 查找所有jsonl文件
    jsonl_files = []
    for pattern in ['**/*.jsonl', '*.jsonl']:
        jsonl_files.extend(input_path.glob(pattern))
    
    if not jsonl_files:
        print(f"在目录 {input_dir} 中未找到jsonl文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件")
    
    # 处理每个文件
    for jsonl_file in jsonl_files:
        # 计算相对路径，保持目录结构
        relative_path = jsonl_file.relative_to(input_path)
        
        # 创建输出文件路径，保持原有的目录结构
        output_file = output_path / relative_path
        
        # 如果输出文件已存在，跳过
        if output_file.exists():
            print(f"跳过已存在的文件: {output_file}")
            continue
        
        filter_jsonl_file(str(jsonl_file), str(output_file))

def main():
    input_dir = "/home/yueyulin/data/voxbox_wids_tokens_with_properties"
    output_dir = "/home/yueyulin/data/voxbox_wids_tokens_filtered"
    
    print("开始过滤jsonl文件，只保留程序需要的字段...")
    print(f"需要的字段: {REQUIRED_FIELDS}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    process_directory(input_dir, output_dir)
    
    print("所有文件处理完成!")

if __name__ == "__main__":
    main() 