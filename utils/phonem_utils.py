import eng_to_ipa as ipa
from pypinyin import pinyin, lazy_pinyin, Style
import jieba3
import random
import re
tokenizer = jieba3.jieba3(model="large") 
def convert_to_ipa(text,lang='en'):
    if lang == 'en':
        return ipa.convert(text)
    elif lang == 'zh':
        return lazy_pinyin(text, style=Style.TONE3)
    else:
        raise ValueError(f"Unsupported language: {lang}")

def ramdomly_mark_phonem(text, lang='en', min_mark=1, max_mark=None):
    global tokenizer
    """
    随机标记音标函数
    :param text: 输入文本
    :param lang: 语言类型 ('en' 或 'zh')
    :param min_mark: 最少标记的单词数量，不能小于1
    :param max_mark: 最多标记的单词数量，不能大于切分后单词总量
    :return: 标记了音标的文本
    """
    if lang == 'en':
        # 按空格分词
        words = text.split()
        if not words:
            return text
        
        # 验证 min_mark 和 max_mark 参数
        if min_mark < 1:
            raise ValueError("min_mark 不能小于 1")
        
        if max_mark is None:
            max_mark = len(words)
        elif max_mark > len(words):
            raise ValueError(f"max_mark ({max_mark}) 不能大于切分后单词总量 ({len(words)})")
        
        if min_mark > max_mark:
            raise ValueError(f"min_mark ({min_mark}) 不能大于 max_mark ({max_mark})")
        
        # 随机决定要标记的单词数量
        mark_count = random.randint(min_mark, max_mark)
        
        # 随机选择要标记的单词索引
        selected_indices = random.sample(range(len(words)), mark_count)
        
        # 对选中的单词进行标记
        for index in selected_indices:
            selected_word = words[index]
            ipa_result = convert_to_ipa(selected_word, 'en')
            words[index] = f"SPCT_48{selected_word}SPCT_49{ipa_result}SPCT_50"
        
        return ' '.join(words)
    
    elif lang == 'zh':
        # 使用 jieba 分词
        words = tokenizer.cut_text(text)
        if not words:
            return text
        
        # 验证 min_mark 和 max_mark 参数
        if min_mark < 1:
            raise ValueError("min_mark 不能小于 1")
        
        if max_mark is None:
            max_mark = len(words)
        elif max_mark > len(words):
            raise ValueError(f"max_mark ({max_mark}) 不能大于切分后单词总量 ({len(words)})")
        
        if min_mark > max_mark:
            raise ValueError(f"min_mark ({min_mark}) 不能大于 max_mark ({max_mark})")
        
        # 随机决定要标记的单词数量
        mark_count = random.randint(min_mark, max_mark)
        
        # 随机选择要标记的单词索引
        selected_indices = random.sample(range(len(words)), mark_count)
        
        # 对选中的单词进行标记
        for index in selected_indices:
            selected_word = words[index]
            ipa_result = convert_to_ipa(selected_word, 'zh')
            # 将列表连接成字符串
            ipa_result = ''.join(ipa_result)
            words[index] = f"SPCT_48{selected_word}SPCT_49{ipa_result}SPCT_50"
        
        return ''.join(words)
    
    else:
        raise ValueError(f"Unsupported language: {lang}")

if __name__ == '__main__':
    print(convert_to_ipa('hello', 'en'))
    print(convert_to_ipa('你好', 'zh'))
    
    # 测试 randomly_mark_phonem 函数
    print("英文测试:")
    eng_str = "Hello world how are you"
    print("原始文本:", eng_str)
    print("默认范围标记 (1-5个单词):", ramdomly_mark_phonem(eng_str, 'en'))
    print("标记1-2个单词:", ramdomly_mark_phonem(eng_str, 'en', min_mark=1, max_mark=2))
    print("标记2-3个单词:", ramdomly_mark_phonem(eng_str, 'en', min_mark=2, max_mark=3))
    print("标记3-4个单词:", ramdomly_mark_phonem(eng_str, 'en', min_mark=3, max_mark=4))
    
    print("\n中文测试:")
    chinese_str = "你好世界今天天气很好"
    print("原始文本:", chinese_str)
    print("默认范围标记 (1-6个单词):", ramdomly_mark_phonem(chinese_str, 'zh'))
    print("标记1-2个单词:", ramdomly_mark_phonem(chinese_str, 'zh', min_mark=1, max_mark=2))
    print("标记2-3个单词:", ramdomly_mark_phonem(chinese_str, 'zh', min_mark=2, max_mark=3))
    print("标记3-4个单词:", ramdomly_mark_phonem(chinese_str, 'zh', min_mark=3, max_mark=4))

    print('\n--------------------------------')
    from transformers import AutoTokenizer
    model_path = '/home/yueyulin/models/rwkv7-0.4B-g1-respark-epoch9-vocab_100/'
    rwkv_tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    marked_chinese_str = ramdomly_mark_phonem(chinese_str, 'zh', min_mark=1, max_mark=3)
    print("中文标记结果:", marked_chinese_str)
    print("中文tokenize结果:", rwkv_tokenizer.encode(marked_chinese_str))
    
    marked_eng_str = ramdomly_mark_phonem(eng_str, 'en', min_mark=1, max_mark=3)
    print("英文标记结果:", marked_eng_str)
    print("英文tokenize结果:", rwkv_tokenizer.encode(marked_eng_str))
    print('--------------------------------')