#!/usr/bin/env python3
"""
文本处理工具
用于处理特殊格式的文本转换
"""

import re
from typing import List, Tuple


def parse_emotion_and_text(text: str) -> str:
    """
    将特殊格式的文本转换为 SPCT 标记格式
    
    输入格式: [intimate, breathy, pleased] Oh, <moan> it feels so good when your fingers spread apart. <moan> Deeper
    输出格式: SPCT_0 intimate, breathy, pleased SPCT_1 Oh, SPCT_2 moan SPCT_3 it feels so good when your fingers spread apart. SPCT_2 moan SPCT_3 Deeper
    
    Args:
        text: 输入的原始文本
        
    Returns:
        转换后的 SPCT 标记文本
    """
    if not text or not isinstance(text, str):
        return text
    
    # 定义正则表达式模式
    emotion_pattern = r'\[([^\]]+)\]'  # 匹配 [emotion1, emotion2, ...]
    moan_pattern = r'<([^>]+)>'       # 匹配 <moan> 等标签
    
    result_parts = []
    current_pos = 0
    
    # 查找所有匹配项
    matches = []
    
    # 查找情绪标签
    for match in re.finditer(emotion_pattern, text):
        matches.append((match.start(), match.end(), 'emotion', match.group(1)))
    
    # 查找 moan 等标签
    for match in re.finditer(moan_pattern, text):
        matches.append((match.start(), match.end(), 'tag', match.group(1)))
    
    # 按位置排序
    matches.sort(key=lambda x: x[0])
    
    # 处理每个匹配项
    for start, end, match_type, content in matches:
        # 添加匹配项之前的文本
        if start > current_pos:
            text_before = text[current_pos:start].strip()
            if text_before:
                result_parts.append(f"SPCT_3 {text_before}")
        
        # 处理匹配项
        if match_type == 'emotion':
            result_parts.append(f"SPCT_0 {content} SPCT_1")
        elif match_type == 'tag':
            result_parts.append(f"SPCT_2 {content} SPCT_3")
        
        current_pos = end
    
    # 添加剩余的文本
    if current_pos < len(text):
        remaining_text = text[current_pos:].strip()
        if remaining_text:
            result_parts.append(f"SPCT_3 {remaining_text}")
    
    # 合并结果
    return ' '.join(result_parts)


def parse_emotion_and_text_v2(text: str) -> str:
    """
    改进版本的文本转换函数，处理更复杂的情况
    
    Args:
        text: 输入的原始文本
        
    Returns:
        转换后的 SPCT 标记文本
    """
    if not text or not isinstance(text, str):
        return text
    text = text.replace("\"", "")
    # 移除首尾空白
    text = text.strip()
    
    result_parts = []
    
    # 使用更精确的正则表达式
    # 匹配情绪标签：[...]
    emotion_regex = r'^\[([^\]]+)\]\s*(.*)$'
    emotion_match = re.match(emotion_regex, text)
    
    if emotion_match:
        emotions = emotion_match.group(1)
        remaining_text = emotion_match.group(2)
        
        # 添加情绪标签，使用 SPCT_0 和 SPCT_1 形成闭合括号
        result_parts.append(f"SPCT_0 {emotions} SPCT_1")
        
        # 处理剩余文本
        if remaining_text:
            result_parts.extend(process_remaining_text(remaining_text))
    else:
        # 没有情绪标签，直接处理文本
        result_parts.extend(process_remaining_text(text))
    
    return ' '.join(result_parts)


def process_remaining_text(text: str) -> List[str]:
    """
    处理剩余文本中的标签和普通文本
    
    Args:
        text: 要处理的文本
        
    Returns:
        处理后的文本片段列表
    """
    parts = []
    
    # 分割文本，保持标签和普通文本分离
    # 使用正则表达式分割，但保留分隔符
    split_pattern = r'(<[^>]+>)'
    segments = re.split(split_pattern, text)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        if segment.startswith('<') and segment.endswith('>'):
            # 这是一个标签，使用 SPCT_2 和 SPCT_3 形成闭合包
            tag_content = segment[1:-1]  # 移除 < >
            parts.append(f"SPCT_2 {tag_content} SPCT_3")
        else:
            # 这是普通文本，直接添加，不添加 SPCT_3 前缀
            if segment:
                parts.append(segment)
    
    return parts


def test_conversion():
    """测试转换函数"""
    test_cases = [
        "[intimate, breathy, pleased] Oh, <moan> it feels so good when your fingers spread apart. <moan> Deeper",
        "[happy, excited] Hello world!",
        "<moan> Just a moan",
        "Plain text without any tags",
        "[sad] <cry> I'm feeling down <cry> very sad",
        "[angry, frustrated] <growl> How dare you! <growl>",
        "[neutral] This is a <pause> test <pause> sentence.",
        "",  # 空字符串
        None,  # None 值
    ]
    
    expected_results = [
        "SPCT_0 intimate, breathy, pleased SPCT_1 Oh, SPCT_2 moan SPCT_3 it feels so good when your fingers spread apart. SPCT_2 moan SPCT_3 Deeper",
        "SPCT_0 happy, excited SPCT_1 Hello world!",
        "SPCT_2 moan SPCT_3 Just a moan",
        "Plain text without any tags",
        "SPCT_0 sad SPCT_1 SPCT_2 cry SPCT_3 I'm feeling down SPCT_2 cry SPCT_3 very sad",
        "SPCT_0 angry, frustrated SPCT_1 SPCT_2 growl SPCT_3 How dare you!",
        "SPCT_0 neutral SPCT_1 This is a SPCT_2 pause SPCT_3 test SPCT_2 pause SPCT_3 sentence.",
        "",
        None,
    ]
    
    print("测试文本转换函数")
    print("=" * 60)
    
    for i, (test_case, expected) in enumerate(zip(test_cases, expected_results), 1):
        print(f"\n测试用例 {i}:")
        print(f"输入: {test_case}")
        
        # 测试两个版本
        result_v1 = parse_emotion_and_text(test_case)
        result_v2 = parse_emotion_and_text_v2(test_case)
        
        print(f"输出 v1: {result_v1}")
        print(f"输出 v2: {result_v2}")
        print(f"预期: {expected}")
        
        # 检查结果是否一致
        if result_v1 == result_v2:
            print("✅ 两个版本结果一致")
        else:
            print("⚠️  两个版本结果不同")
        
        # 检查是否符合预期
        if result_v2 == expected:
            print("✅ 结果符合预期")
        else:
            print("❌ 结果不符合预期")


def batch_process_texts(texts: List[str], use_v2: bool = True) -> List[str]:
    """
    批量处理文本列表
    
    Args:
        texts: 要处理的文本列表
        use_v2: 是否使用改进版本
        
    Returns:
        处理后的文本列表
    """
    if use_v2:
        return [parse_emotion_and_text_v2(text) for text in texts]
    else:
        return [parse_emotion_and_text(text) for text in texts]


if __name__ == '__main__':
    # 运行测试
    test_conversion()
    
    # 示例使用
    print("\n" + "=" * 60)
    print("示例使用:")
    
    example_text = "[intimate, breathy, pleased] Oh, <moan> it feels so good when your fingers spread apart. <moan> Deeper"
    result = parse_emotion_and_text_v2(example_text)
    expected = "SPCT_0 intimate, breathy, pleased SPCT_1 Oh, SPCT_2 moan SPCT_3 it feels so good when your fingers spread apart. SPCT_2 moan SPCT_3 Deeper"
    
    print(f"原始文本: {example_text}")
    print(f"转换结果: {result}")
    print(f"预期结果: {expected}")
    print(f"结果正确: {'✅' if result == expected else '❌'}")
    
    # 批量处理示例
    print("\n批量处理示例:")
    texts = [
        "[happy] Hello <smile> world!",
        "[sad] <cry> Goodbye",
        "Plain text"
    ]
    
    expected_results = [
        "SPCT_0 happy SPCT_1 Hello SPCT_2 smile SPCT_3 world!",
        "SPCT_0 sad SPCT_1 SPCT_2 cry SPCT_3 Goodbye",
        "Plain text"
    ]
    
    results = batch_process_texts(texts)
    for original, converted, expected in zip(texts, results, expected_results):
        print(f"原始: {original}")
        print(f"转换: {converted}")
        print(f"预期: {expected}")
        print(f"正确: {'✅' if converted == expected else '❌'}")
        print() 