SPEED_MAP = {
    "very_slow": "SPCT_1",
    "slow": "SPCT_2",
    "medium": "SPCT_3",
    "fast": "SPCT_4",
    "very_fast": "SPCT_5",
}

PITCH_MAP = {
    "low_pitch": "SPCT_6",
    "medium_pitch": "SPCT_7", 
    "high_pitch": "SPCT_8",
    "very_high_pitch": "SPCT_9",
}

AGE_MAP = {
    "child": "SPCT_13",
    "teenager": "SPCT_14",
    "youth-adult": "SPCT_15",
    "middle-aged": "SPCT_16",
    "elderly": "SPCT_17",
}

GENDER_MAP = {
    "male": "SPCT_18",
    "female": "SPCT_19",
    "unknown": "SPCT_20",
}

EMOTION_MAP = {
    "UNKNOWN": "SPCT_21",
    "NEUTRAL": "SPCT_22",
    "ANGRY": "SPCT_23",
    "HAPPY": "SPCT_24",
    "SAD": "SPCT_25",
    "FEARFUL": "SPCT_26",
    "DISGUSTED": "SPCT_27",
    "SURPRISED": "SPCT_28",
    "SARCASTIC": "SPCT_29",
    "EXCITED": "SPCT_30",
    "SLEEPY": "SPCT_31",
    "CONFUSED": "SPCT_32",
    "EMPHASIS": "SPCT_33",
    "LAUGHING": "SPCT_34",
    "SINGING": "SPCT_35",
    "WORRIED": "SPCT_36",
    "WHISPER": "SPCT_37", 
    "ANXIOUS": "SPCT_38",
    "NO-AGREEMENT": "SPCT_39",
    "APOLOGETIC": "SPCT_40",
    "CONCERNED": "SPCT_41",
    "ENUNCIATED": "SPCT_42",
    "ASSERTIVE": "SPCT_43",
    "ENCOURAGING": "SPCT_44",
    "CONTEMPT": "SPCT_45",
}

# 注意：这里有两个GENDER_MAP定义，第二个会覆盖第一个
# 第一个定义包含了"unknown"，第二个只包含"female"和"male"
# 建议使用第二个定义，因为它更简洁且符合实际使用场景
GENDER_MAP = {
    "female": "SPCT_46",
    "male": "SPCT_47"
}

def convert_standard_properties_to_tokens(age: str, gender: str, emotion: str, pitch: str, speed: str) -> list:
    age_token = AGE_MAP[age.lower()]
    gender_token = GENDER_MAP[gender.lower()]
    emotion_token = EMOTION_MAP[emotion.upper()]
    pitch_token = PITCH_MAP[pitch.lower()]
    speed_token = SPEED_MAP[speed.lower()]
    return "SPCT_0"+age_token+gender_token+emotion_token+pitch_token+speed_token

def convert_properties_to_tokens(age: str, gender: str, emotion: str, pitch: float, speed: float) -> list:
    age_token = AGE_MAP[age.lower()]
    gender_token = GENDER_MAP[gender.lower()]
    emotion_token = EMOTION_MAP[emotion.upper()]
    pitch_token = PITCH_MAP[classify_pitch(pitch, gender.lower(), age.lower())]
    speed_token = SPEED_MAP[classify_speed(speed)]
    return "SPCT_0"+age_token+gender_token+emotion_token+pitch_token+speed_token

def classify_speed(speed: float) -> str:
    if speed <= 3.5:
        return "very_slow"
    elif 3.5 < speed < 4.0:
        return "slow"
    elif 4.0 < speed <= 4.5:
        return "medium"
    elif 4.5 < speed <= 5.0:
        return "fast"
    else: # speed >= 5.0
        return "very_fast"
def classify_pitch(pitch: float, gender: str, age: str) -> str:
    """
    根据性别和年龄重新划分pitch区间
    基于统计结果：
    - female: 平均212.08, 中位数208.76, 25%分位数187.40, 75%分位数232.08
    - male: 平均136.22, 中位数129.65, 25%分位数113.76, 75%分位数151.42
    """
    gender = gender.lower()
    age = age.lower()
    
    # 女性分类
    if gender == "female":
        if age == "child":
            # Child: 平均280.12, 中位数279.34, 范围216.91-324.25
            if pitch < 250:
                return "low_pitch"
            elif pitch < 290:
                return "medium_pitch"
            else:
                return "high_pitch"
        elif age == "teenager":
            # Teenager: 平均240.61, 中位数238.43, 25%分位数207.54, 75%分位数270.12
            if pitch < 208:
                return "low_pitch"
            elif pitch < 238:
                return "medium_pitch"
            elif pitch < 270:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "youth-adult":
            # Youth-Adult: 平均213.26, 中位数210.99, 25%分位数190.81, 75%分位数232.24
            if pitch < 191:
                return "low_pitch"
            elif pitch < 211:
                return "medium_pitch"
            elif pitch < 232:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "middle-aged":
            # Middle-aged: 平均197.68, 中位数195.01, 25%分位数176.34, 75%分位数215.22
            if pitch < 176:
                return "low_pitch"
            elif pitch < 195:
                return "medium_pitch"
            elif pitch < 215:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "elderly":
            # Elderly: 平均194.91, 中位数189.90, 25%分位数170.42, 75%分位数213.41
            if pitch < 170:
                return "low_pitch"
            elif pitch < 190:
                return "medium_pitch"
            elif pitch < 213:
                return "high_pitch"
            else:
                return "very_high_pitch"
        else:
            # 默认女性分类
            if pitch < 187:
                return "low_pitch"
            elif pitch < 209:
                return "medium_pitch"
            elif pitch < 232:
                return "high_pitch"
            else:
                return "very_high_pitch"
    
    # 男性分类
    elif gender == "male":
        if age == "teenager":
            # Teenager: 平均150.93, 中位数142.50, 25%分位数121.47, 75%分位数165.55
            if pitch < 121:
                return "low_pitch"
            elif pitch < 143:
                return "medium_pitch"
            elif pitch < 166:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "youth-adult":
            # Youth-Adult: 平均137.17, 中位数130.92, 25%分位数114.70, 75%分位数153.18
            if pitch < 115:
                return "low_pitch"
            elif pitch < 131:
                return "medium_pitch"
            elif pitch < 153:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "middle-aged":
            # Middle-aged: 平均132.33, 中位数125.30, 25%分位数110.31, 75%分位数146.55
            if pitch < 110:
                return "low_pitch"
            elif pitch < 125:
                return "medium_pitch"
            elif pitch < 147:
                return "high_pitch"
            else:
                return "very_high_pitch"
        elif age == "elderly":
            # Elderly: 平均132.62, 中位数128.42, 25%分位数114.69, 75%分位数141.57
            if pitch < 115:
                return "low_pitch"
            elif pitch < 128:
                return "medium_pitch"
            elif pitch < 142:
                return "high_pitch"
            else:
                return "very_high_pitch"
        else:
            # 默认男性分类
            if pitch < 114:
                return "low_pitch"
            elif pitch < 130:
                return "medium_pitch"
            elif pitch < 151:
                return "high_pitch"
            else:
                return "very_high_pitch"
    
    # 未知性别，使用通用分类
    else:
        if pitch < 130:
            return "low_pitch"
        elif pitch < 180:
            return "medium_pitch"
        elif pitch < 220:
            return "high_pitch"
        else:
            return "very_high_pitch"