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
    if pitch <= 118.09:
        return "very_low_pitch"
    elif 118.09 < pitch <= 139.34:
        return "low_pitch"
    elif 139.34 < pitch <= 172.52:
        return "medium_pitch"
    elif 172.52 < pitch <= 210.07:
        return "high_pitch"
    else:# pitch >= 210.07
        return "very_high_pitch"