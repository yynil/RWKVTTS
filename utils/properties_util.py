def classify_speed(speed: float) -> str:
    """根据经验和领域知识划分语速为5档。"""
    if speed < 3.0:
        return "非常慢"
    elif 3.0 <= speed < 4.0:
        return "慢"
    elif 4.0 <= speed < 5.5:
        return "中等"
    elif 5.5 <= speed < 6.5:
        return "快"
    else: # speed >= 6.5
        return "非常快"
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
            return "低 (儿童)"
        elif 250 <= pitch < 300:
            return "中等 (儿童)"
        elif 300 <= pitch < 350:
            return "高 (儿童)"
        elif 350 <= pitch < 400:
            return "非常高 (儿童)"
        else: # pitch >= 400
            return "极高 (儿童)" # 为儿童添加一个更高的档位

    # 成年人及青少年 (Teenager, Youth-Adult, Middle-aged)
    elif age in ["teenager", "youth-adult", "middle-aged"]:
        if gender == "female":
            if pitch < 160:
                return "非常低 (女性)"
            elif 160 <= pitch < 190:
                return "低 (女性)"
            elif 190 <= pitch < 230:
                return "中等 (女性)"
            elif 230 <= pitch < 270:
                return "高 (女性)"
            else: # pitch >= 270
                return "非常高 (女性)"
        elif gender == "male":
            if pitch < 90:
                return "非常低 (男性)"
            elif 90 <= pitch < 110:
                return "低 (男性)"
            elif 110 <= pitch < 140:
                return "中等 (男性)"
            elif 140 <= pitch < 170:
                return "高 (男性)"
            else: # pitch >= 170
                return "非常高 (男性)"
        else:
            # 未知性别，使用通用成年人范围 (不建议，应尽量获取性别)
            if pitch < 100:
                return "非常低 (未知性别)"
            elif 100 <= pitch < 150:
                return "低 (未知性别)"
            elif 150 <= pitch < 200:
                return "中等 (未知性别)"
            elif 200 <= pitch < 250:
                return "高 (未知性别)"
            else:
                return "非常高 (未知性别)"

    # 老年人 (Elderly)
    elif age == "elderly":
        if gender == "female":
            # 老年女性音高可能略有下降
            if pitch < 150:
                return "非常低 (老年女性)"
            elif 150 <= pitch < 180:
                return "低 (老年女性)"
            elif 180 <= pitch < 220:
                return "中等 (老年女性)"
            elif 220 <= pitch < 260:
                return "高 (老年女性)"
            else: # pitch >= 260
                return "非常高 (老年女性)"
        elif gender == "male":
            # 老年男性音高可能略有上升
            if pitch < 100:
                return "非常低 (老年男性)"
            elif 100 <= pitch < 120:
                return "低 (老年男性)"
            elif 120 <= pitch < 150:
                return "中等 (老年男性)"
            elif 150 <= pitch < 180:
                return "高 (老年男性)"
            else: # pitch >= 180
                return "非常高 (老年男性)"
        else:
            # 未知性别老年人 (不建议，应尽量获取性别)
            if pitch < 100:
                return "非常低 (未知性别老年人)"
            elif 100 <= pitch < 140:
                return "低 (未知性别老年人)"
            elif 140 <= pitch < 180:
                return "中等 (未知性别老年人)"
            elif 180 <= pitch < 220:
                return "高 (未知性别老年人)"
            else:
                return "非常高 (未知性别老年人)"
    else:
        # 未知年龄类型
        print(f"Warning: Unknown age category '{age}'. Using default ranges based on gender if available.")
        # 回退到只根据性别分类（如果你希望这样做）
        if gender == "female":
            if pitch < 160: return "非常低 (女性, 未知年龄)"
            elif 160 <= pitch < 190: return "低 (女性, 未知年龄)"
            elif 190 <= pitch < 230: return "中等 (女性, 未知年龄)"
            elif 230 <= pitch < 270: return "高 (女性, 未知年龄)"
            else: return "非常高 (女性, 未知年龄)"
        elif gender == "male":
            if pitch < 90: return "非常低 (男性, 未知年龄)"
            elif 90 <= pitch < 110: return "低 (男性, 未知年龄)"
            elif 110 <= pitch < 140: return "中等 (男性, 未知年龄)"
            elif 140 <= pitch < 170: return "高 (男性, 未知年龄)"
            else: return "非常高 (男性, 未知年龄)"
        else:
            return "无法分类 (未知性别和年龄)"