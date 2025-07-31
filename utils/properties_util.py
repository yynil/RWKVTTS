SPEED_MAP = {
    "very_slow": "SPCT_1",
    "slow": "SPCT_2",
    "medium": "SPCT_3",
    "fast": "SPCT_4",
    "very_fast": "SPCT_5",
}

PITCH_MAP = {
    "very_low_pitch": "SPCT_6",
    "low_pitch": "SPCT_7",
    "medium_pitch": "SPCT_8",
    "high_pitch": "SPCT_9",
    "very_high_pitch": "SPCT_10",
    "extreme_high_pitch": "SPCT_11",
    "unknown_pitch": "SPCT_12",
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

GENDER_MAP = {
    "female": "SPCT_46",
    "male": "SPCT_47"
}

def convert_properties_to_tokens(age: str, gender: str, emotion: str, pitch: float, speed: float) -> list:
    age_token = AGE_MAP[age]
    gender_token = GENDER_MAP[gender]
    emotion_token = EMOTION_MAP[emotion]
    pitch_token = classify_pitch(pitch, gender, age)
    speed_token = classify_speed(speed)
    return "SPCT_1 "+age_token+' '+gender_token+' '+emotion_token+' '+pitch_token+' '+speed_token

def classify_speed(speed: float) -> str:
    """根据经验和领域知识划分语速为5档。"""
    if speed < 3.0:
        return "very_slow"
    elif 3.0 <= speed < 4.0:
        return "slow"
    elif 4.0 <= speed < 5.5:
        return "medium"
    elif 5.5 <= speed < 6.5:
        return "fast"
    else: # speed >= 6.5
        return "very_fast"
def classify_pitch(pitch: float, gender: str, age: str) -> str:
    """
    根据经验和领域知识，并考虑性别和年龄划分音高为5档。
    年龄取值: {"Child": 0, "Teenager": 1, "Youth-Adult": 2, "Middle-aged": 3, "Elderly": 4}
    """
    gender = gender.lower()
    age = age.lower()

    # 儿童 (Child)
    if age == "child":
        if pitch < 250:
            return "low_pitch"
        elif 250 <= pitch < 300:
            return "medium_pitch"
        elif 300 <= pitch < 350:
            return "high_pitch"
        elif 350 <= pitch < 400:
            return "very_high_pitch"
        else: # pitch >= 400
            return "extreme_high_pitch" # 为儿童添加一个更高的档位

    # 成年人及青少年 (Teenager, Youth-Adult, Middle-aged)
    elif age in ["teenager", "youth-adult", "middle-aged"]:
        if gender == "female":
            if pitch < 160:
                return "very_low_pitch"
            elif 160 <= pitch < 190:
                return "low_pitch"
            elif 190 <= pitch < 230:
                return "medium_pitch"
            elif 230 <= pitch < 270:
                return "high_pitch"
            else: # pitch >= 270
                return "very_high_pitch"
        elif gender == "male":
            if pitch < 90:
                return "very_low_pitch"
            elif 90 <= pitch < 110:
                return "low_pitch"
            elif 110 <= pitch < 140:
                return "medium_pitch"
            elif 140 <= pitch < 170:
                return "high_pitch"
            else: # pitch >= 170
                return "very_high_pitch"
        else:
            # 未知性别，使用通用成年人范围 (不建议，应尽量获取性别)
            if pitch < 100:
                return "very_low_pitch"
            elif 100 <= pitch < 150:
                return "low_pitch"
            elif 150 <= pitch < 200:
                return "medium_pitch"
            elif 200 <= pitch < 250:
                return "high_pitch"
            else:
                return "very_high_pitch"

    # 老年人 (Elderly)
    elif age == "elderly":
        if gender == "female":
            # 老年女性音高可能略有下降
            if pitch < 150:
                return "very_low_pitch"
            elif 150 <= pitch < 180:
                return "low_pitch"
            elif 180 <= pitch < 220:
                return "medium_pitch"
            elif 220 <= pitch < 260:
                return "high_pitch"
            else: # pitch >= 260
                return "very_high_pitch"
        elif gender == "male":
            # 老年男性音高可能略有上升
            if pitch < 100:
                return "very_low_pitch"
            elif 100 <= pitch < 120:
                return "low_pitch"
            elif 120 <= pitch < 150:
                return "medium_pitch"
            elif 150 <= pitch < 180:
                return "high_pitch"
            else: # pitch >= 180
                return "very_high_pitch"
        else:
            # 未知性别老年人 (不建议，应尽量获取性别)
            if pitch < 100:
                return "very_low_pitch"
            elif 100 <= pitch < 140:
                return "low_pitch"
            elif 140 <= pitch < 180:
                return "medium_pitch"
            elif 180 <= pitch < 220:
                return "high_pitch"
            else:
                return "very_high_pitch"
    else:
        # 未知年龄类型
        print(f"Warning: Unknown age category '{age}'. Using default ranges based on gender if available.")
        # 回退到只根据性别分类（如果你希望这样做）
        if gender == "female":
            if pitch < 160: return "very_low_pitch"
            elif 160 <= pitch < 190: return "low_pitch"
            elif 190 <= pitch < 230: return "medium_pitch"
            elif 230 <= pitch < 270: return "high_pitch"
            else: return "very_high_pitch"
        elif gender == "male":
            if pitch < 90: return "very_low_pitch"
            elif 90 <= pitch < 110: return "low_pitch"
            elif 110 <= pitch < 140: return "medium_pitch"
            elif 140 <= pitch < 170: return "high_pitch"
            else: return "very_high_pitch"
        else:
            return "unknown_pitch"