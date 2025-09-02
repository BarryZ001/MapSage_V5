#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算LoveDA数据集的均值和标准差
用于修正配置文件中的data_preprocessor参数
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_mean_std(image_dir):
    """
    计算数据集中所有图像的均值和标准差。
    """
    # 图像文件列表
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        print(f"错误：在目录 '{image_dir}' 中未找到任何图像文件。")
        return None, None

    # 初始化变量
    pixel_sum = np.zeros(3)  # B, G, R
    pixel_sum_sq = np.zeros(3)
    pixel_count = 0

    print(f"开始计算 {len(image_files)} 张图片的均值和标准差...")
    for img_path in tqdm(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 将图像转换为浮点数以进行精确计算
        img_float = img.astype(np.float64) / 255.0
        
        # 累加像素值和像素值的平方
        pixel_sum += np.sum(img_float, axis=(0, 1))
        pixel_sum_sq += np.sum(img_float ** 2, axis=(0, 1))
        
        # 累加像素总数
        h, w, _ = img.shape
        pixel_count += h * w

    if pixel_count == 0:
        print("错误：无法读取任何有效的图像像素。")
        return None, None

    # 计算均值
    mean = pixel_sum / pixel_count
    # 计算标准差
    std = np.sqrt(pixel_sum_sq / pixel_count - mean ** 2)
    
    # 将0-1范围的值转换回0-255范围
    mean_255 = mean * 255
    std_255 = std * 255
    
    # 通常的顺序是 BGR，但mmseg的配置是RGB，所以我们反转一下
    # 注意：cv2读取的是BGR，所以mean_255[0]是B, [1]是G, [2]是R
    # 配置文件需要RGB顺序，即 [R, G, B]
    mean_rgb = mean_255[::-1]
    std_rgb = std_255[::-1]

    return mean_rgb, std_rgb

def calculate_multiple_dirs(image_dirs):
    """
    计算多个目录中所有图像的均值和标准差。
    """
    all_image_files = []
    for image_dir in image_dirs:
        if os.path.exists(image_dir):
            files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
            all_image_files.extend(files)
            print(f"找到目录: {image_dir} ({len(files)} 张图片)")
        else:
            print(f"目录不存在: {image_dir}")
    
    if not all_image_files:
        print("错误：未找到任何图像文件。")
        return None, None

    # 初始化变量
    pixel_sum = np.zeros(3)  # B, G, R
    pixel_sum_sq = np.zeros(3)
    pixel_count = 0

    print(f"\n开始计算 {len(all_image_files)} 张图片的均值和标准差...")
    for img_path in tqdm(all_image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 将图像转换为浮点数以进行精确计算
        img_float = img.astype(np.float64) / 255.0
        
        # 累加像素值和像素值的平方
        pixel_sum += np.sum(img_float, axis=(0, 1))
        pixel_sum_sq += np.sum(img_float ** 2, axis=(0, 1))
        
        # 累加像素总数
        h, w, _ = img.shape
        pixel_count += h * w

    if pixel_count == 0:
        print("错误：无法读取任何有效的图像像素。")
        return None, None

    # 计算均值
    mean = pixel_sum / pixel_count
    # 计算标准差
    std = np.sqrt(pixel_sum_sq / pixel_count - mean ** 2)
    
    # 将0-1范围的值转换回0-255范围
    mean_255 = mean * 255
    std_255 = std * 255
    
    # 通常的顺序是 BGR，但mmseg的配置是RGB，所以我们反转一下
    # 注意：cv2读取的是BGR，所以mean_255[0]是B, [1]是G, [2]是R
    # 配置文件需要RGB顺序，即 [R, G, B]
    mean_rgb = mean_255[::-1]
    std_rgb = std_255[::-1]

    return mean_rgb, std_rgb

def main():
    # 使用项目中现有的LoveDA验证集数据
    loveda_image_dirs = [
        'images/Val/Rural/images_png',
        'images/Val/Urban/images_png'
    ]
    
    print("使用项目中的LoveDA验证集数据计算均值和标准差...")
    print("注意：这是验证集数据，理论上应该与训练集具有相似的统计特性。")
    print()
    
    mean, std = calculate_multiple_dirs(loveda_image_dirs)
    
    if mean is not None and std is not None:
        print("\n" + "="*60)
        print("计算完成！")
        print("="*60)
        print("请将以下数值更新到您的配置文件中的 `data_preprocessor` 部分：")
        print(f"mean = {mean.tolist()}")
        print(f"std = {std.tolist()}")
        print("\n当前配置文件中使用的是ImageNet的值：")
        print("mean = [123.675, 116.28, 103.53]")
        print("std = [58.395, 57.12, 57.375]")
        print("\n如果LoveDA的值与ImageNet差异较大，这就是问题所在！")
        print("="*60)
    else:
        print("计算失败，请检查数据集路径是否正确。")

if __name__ == "__main__":
    main()