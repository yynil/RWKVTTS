#!/usr/bin/env python3
"""
创建RWKV TTS应用的图标
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """创建一个包含RWKVTTS文字和话筒的图标"""
    # 创建一个128x128的图像
    size = 128
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # 绘制一个圆形背景
    circle_color = (52, 152, 219)  # 蓝色
    draw.ellipse([8, 8, size-8, size-8], fill=circle_color)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        # 尝试使用系统字体
        font_size = 24
        try:
            # macOS系统字体
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                # 尝试其他常见字体
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制RWKVTTS文字
    text = "RWKVTTS"
    text_color = (255, 255, 255)  # 白色
    
    # 计算文字位置，使其居中
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    # 绘制文字阴影（可选）
    shadow_offset = 2
    draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 128), font=font)
    
    # 绘制主文字
    draw.text((x, y), text, fill=text_color, font=font)
    
    # 在底部添加一个话筒符号
    mic_color = (255, 255, 255, 180)  # 半透明白色
    
    # 话筒位置（底部中央）
    mic_x = size // 2 - 15
    mic_y = size - 40
    
    # 绘制话筒的头部（圆形）
    mic_head_radius = 12
    mic_head_x = mic_x + 15
    mic_head_y = mic_y + 8
    draw.ellipse([mic_head_x - mic_head_radius, mic_head_y - mic_head_radius, 
                   mic_head_x + mic_head_radius, mic_head_y + mic_head_radius], 
                  fill=mic_color)
    
    # 绘制话筒的颈部（矩形）
    mic_neck_width = 8
    mic_neck_height = 20
    mic_neck_x = mic_x + 15 - mic_neck_width // 2
    mic_neck_y = mic_y + 8 + mic_head_radius
    draw.rectangle([mic_neck_x, mic_neck_y, 
                    mic_neck_x + mic_neck_width, mic_neck_y + mic_neck_height], 
                   fill=mic_color)
    
    # 绘制话筒的底座（椭圆形）
    mic_base_width = 20
    mic_base_height = 8
    mic_base_x = mic_x + 15 - mic_base_width // 2
    mic_base_y = mic_y + 8 + mic_head_radius + mic_neck_height
    draw.ellipse([mic_base_x, mic_base_y, 
                  mic_base_x + mic_base_width, mic_base_y + mic_base_height], 
                 fill=mic_color)
    
    # 绘制话筒头部的网格纹理（小圆点）
    grid_color = (52, 152, 219, 100)  # 半透明蓝色
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i*i + j*j <= 4:  # 圆形网格
                dot_x = mic_head_x + i * 3
                dot_y = mic_head_y + j * 3
                if (dot_x - mic_head_x)**2 + (dot_y - mic_head_y)**2 <= mic_head_radius**2:
                    draw.ellipse([dot_x - 1, dot_y - 1, dot_x + 1, dot_y + 1], 
                               fill=grid_color)
    
    # 保存图标
    icon_path = "tts_icon.png"
    image.save(icon_path, "PNG")
    print(f"图标已保存到: {icon_path}")
    
    # 转换为ICO格式（Windows）
    try:
        ico_path = "tts_icon.ico"
        image.save(ico_path, "ICO")
        print(f"ICO图标已保存到: {ico_path}")
    except Exception as e:
        print(f"无法保存ICO格式: {e}")
    
    return icon_path

if __name__ == "__main__":
    create_icon()
