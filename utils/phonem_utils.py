import eng_to_ipa as ipa
from pypinyin import pinyin, lazy_pinyin, Style
import jieba3
import random
import re
import threading

# 懒加载的 jieba 分词器（进程内单例，线程安全初始化）
_jieba_tokenizer = None
_jieba_lock = threading.Lock()
from pypinyin import pinyin, Style
from collections import defaultdict
import random
import warnings

# 全局缓存
_pinyin_to_chars_cache = None  # {TONE3拼音: [同音字]}
_all_chinese_chars = None      # 所有有效汉字列表
_word_list = None
def build_pinyin_dict():
    """预加载所有汉字拼音映射，构建倒排索引和汉字列表"""
    global _pinyin_to_chars_cache, _all_chinese_chars
    
    if _pinyin_to_chars_cache is not None:
        return _pinyin_to_chars_cache
    
    _pinyin_to_chars_cache = defaultdict(list)
    _all_chinese_chars = []
    
    for code in range(0x4E00, 0x9FA5 + 1):  # 基本汉字Unicode范围
        char = chr(code)
        try:
            # 获取所有可能的读音（多音字处理）
            pinyin_lists = pinyin(char, style=Style.TONE3, heteronym=True)
            for py_list in pinyin_lists:
                for py in py_list:
                    _pinyin_to_chars_cache[py].append(char)
            _all_chinese_chars.append(char)
        except:
            continue
    
    return _pinyin_to_chars_cache

def get_random_chinese_char():
    """随机返回一个有效汉字（基于pypinyin能处理的字符）"""
    if _all_chinese_chars is None:
        build_pinyin_dict()
    return random.choice(_all_chinese_chars)

def get_chars_by_tone3(pinyin_str, strict=True):
    """根据TONE3拼音返回同音字列表"""
    if _pinyin_to_chars_cache is None:
        build_pinyin_dict()
    if strict:
        return _pinyin_to_chars_cache.get(pinyin_str, [])
    else:
        base_py = pinyin_str.rstrip('12345')
        return [char for py, chars in _pinyin_to_chars_cache.items() 
                if py.startswith(base_py) for char in chars]
def get_random_eng_world():
    global _word_list
    if _word_list is None:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        _word_list = list(spell.word_frequency.dictionary.keys())
    return random.choice(_word_list)
def get_random_char_by_tone3(pinyin_str, strict=True):
    """随机返回一个同音字"""
    chars = get_chars_by_tone3(pinyin_str, strict)
    if not chars:
        warnings.warn(f"未找到拼音为 {pinyin_str} 的汉字")
        return get_random_chinese_char()  # 失败时回退到随机汉字
    return random.choice(chars)

def get_jieba_tokenizer():
    global _jieba_tokenizer
    if _jieba_tokenizer is None:
        with _jieba_lock:
            if _jieba_tokenizer is None:
                _jieba_tokenizer = jieba3.jieba3(model="large")
    return _jieba_tokenizer

def detect_token_lang(token: str) -> str:
    """基于字符集合的简单词级语言检测。返回 'en' 或 'zh'。
    - 含有中文字符则判为 'zh'
    - 含有英文字母则判为 'en'
    - 两者都有时优先中文（更适合中文句子里的中英混排）
    - 都没有则回退为 'en'
    """
    if not token:
        return 'en'
    has_zh = re.search(r"[\u4e00-\u9fff]", token) is not None
    has_en = re.search(r"[A-Za-z]", token) is not None
    if has_zh and not has_en:
        return 'zh'
    if has_en and not has_zh:
        return 'en'
    if has_zh and has_en:
        return 'zh'
    return 'en'

def convert_to_ipa_str(text: str, lang: str) -> str:
    """调用 convert_to_ipa 并统一返回字符串，同时对异常进行兜底。"""
    try:
        result = convert_to_ipa(text, lang)
        return result
    except Exception:
        # 任何异常时，返回原词，避免流程中断
        return text
def convert_to_ipa(text,lang='en'):
    if lang == 'en':
        return ipa.convert(text)
    elif lang == 'zh':
        return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    else:
        raise ValueError(f"Unsupported language: {lang}")

def ramdomly_mark_phonem(text, lang='en', min_mark=1, max_mark=None,wrong_word_ratio=0.5):
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
            max_mark = len(words)
        
        if min_mark > max_mark:
            raise ValueError(f"min_mark ({min_mark}) 不能大于 max_mark ({max_mark})")
        
        # 随机决定要标记的单词数量
        mark_count = random.randint(min_mark, max_mark)
        
        # 随机选择要标记的单词索引
        selected_indices = random.sample(range(len(words)), mark_count)
        
        # 对选中的单词进行标记（按词检测语言）
        for index in selected_indices:
            selected_word = words[index]
            token_lang = detect_token_lang(selected_word)
            ipa_result = convert_to_ipa_str(selected_word, token_lang)
            if token_lang == 'zh':
                #split word to characters
                zh_chars = list(selected_word)
                #ipa_result is already a list
                if len(ipa_result) != len(zh_chars):
                    words[index] = selected_word # For cases where the word is not split correctly
                else:
                    marked_word = ''
                    for i in range(len(zh_chars)):
                        if random.random() < wrong_word_ratio:
                            wrong_char = get_random_chinese_char()
                            marked_word += f"SPCT_48{wrong_char}SPCT_49{ipa_result[i]}SPCT_50"
                        else:
                            marked_word += f"SPCT_48{zh_chars[i]}SPCT_49{ipa_result[i]}SPCT_50"
                    words[index] = marked_word
            else:
                if random.random() < wrong_word_ratio:
                    wrong_word = get_random_eng_world()
                    words[index] = f"SPCT_48{wrong_word}SPCT_49{ipa_result}SPCT_50"
                else:
                    words[index] = f"SPCT_48{selected_word}SPCT_49{ipa_result}SPCT_50"
        
        return ' '.join(words)
    
    elif lang == 'zh':
        # 使用 jieba 分词（懒加载）
        words = get_jieba_tokenizer().cut_text(text)
        if not words:
            return text
        
        # 验证 min_mark 和 max_mark 参数
        if min_mark < 1:
            raise ValueError("min_mark 不能小于 1")
        
        if max_mark is None:
            max_mark = len(words)
        elif max_mark > len(words):
            max_mark = len(words)
        
        if min_mark > max_mark:
            raise ValueError(f"min_mark ({min_mark}) 不能大于 max_mark ({max_mark})")
        
        # 随机决定要标记的单词数量
        mark_count = random.randint(min_mark, max_mark)
        
        # 随机选择要标记的单词索引
        selected_indices = random.sample(range(len(words)), mark_count)
        
        # 对选中的单词进行标记（按词检测语言）
        for index in selected_indices:
            selected_word = words[index]
            token_lang = detect_token_lang(selected_word)
            ipa_result = convert_to_ipa_str(selected_word, token_lang)
            if token_lang == 'zh':
                #split word to characters
                zh_chars = list(selected_word)
                #ipa_result is already a list
                if len(ipa_result) != len(zh_chars):
                    words[index] = selected_word # For cases where the word is not split correctly
                else:
                    marked_word = ''
                    for i in range(len(zh_chars)):
                        if random.random() < wrong_word_ratio:
                            wrong_char = get_random_chinese_char()
                            marked_word += f"SPCT_48{wrong_char}SPCT_49{ipa_result[i]}SPCT_50"
                        else:
                            marked_word += f"SPCT_48{zh_chars[i]}SPCT_49{ipa_result[i]}SPCT_50"
                    words[index] = marked_word
            else:
                if random.random() < wrong_word_ratio:
                    wrong_word = get_random_eng_world()
                    words[index] = f"SPCT_48{wrong_word}SPCT_49{ipa_result}SPCT_50"
                else:
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
    chinese_str = "Wow,你好世界今天天气很好"
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

    str = "叶公好龙"
    print(ramdomly_mark_phonem(str, 'zh', min_mark=1, max_mark=1))