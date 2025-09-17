#!/usr/bin/env python3
"""
犀牛鸟项目基本使用示例
Rhino Bird Project Basic Usage Example

这个示例展示了如何使用犀牛鸟项目的核心功能
This example demonstrates how to use the core functionality of the Rhino Bird Project
"""

import sys
import os

# 添加src目录到Python路径
# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import RhinoBird, create_rhino_bird
from __init__ import get_info, get_version

def main():
    """主函数 / Main function"""
    print("🦏🐦 欢迎使用犀牛鸟项目! / Welcome to Rhino Bird Project!")
    print("=" * 60)
    
    # 显示项目信息 / Display project information
    info = get_info()
    print(f"项目名称 / Project Name: {info['name']}")
    print(f"版本 / Version: {info['version']}")
    print(f"作者 / Author: {info['author']}")
    print(f"描述 / Description: {info['description']}")
    print()
    
    # 创建犀牛鸟实例 / Create Rhino Bird instance
    print("📝 创建犀牛鸟实例... / Creating Rhino Bird instance...")
    my_bird = create_rhino_bird("我的犀牛鸟")
    print(f"✅ 创建成功: {my_bird.name}")
    print(f"📊 初始状态 / Initial status: {my_bird.get_status()}")
    print()
    
    # 让犀牛鸟飞翔 / Make the Rhino Bird fly
    print("🚀 让犀牛鸟起飞... / Making the Rhino Bird take off...")
    fly_message = my_bird.fly()
    print(f"✈️  {fly_message}")
    print(f"📊 当前状态 / Current status: {my_bird.get_status()}")
    print()
    
    # 让犀牛鸟着陆 / Make the Rhino Bird land
    print("🛬 让犀牛鸟着陆... / Making the Rhino Bird land...")
    land_message = my_bird.land()
    print(f"🏁 {land_message}")
    print(f"📊 当前状态 / Current status: {my_bird.get_status()}")
    print()
    
    # 创建多个犀牛鸟进行演示 / Create multiple Rhino Birds for demonstration
    print("🎪 多犀牛鸟演示 / Multiple Rhino Birds demonstration:")
    print("-" * 40)
    
    birds = [
        create_rhino_bird("小红"),
        create_rhino_bird("小蓝"),
        create_rhino_bird("小绿")
    ]
    
    for i, bird in enumerate(birds, 1):
        print(f"{i}. {bird.name}: {bird.fly()}")
    
    print()
    print("🎉 示例演示完成! / Example demonstration completed!")
    print("感谢使用犀牛鸟项目! / Thank you for using Rhino Bird Project!")

if __name__ == "__main__":
    main()