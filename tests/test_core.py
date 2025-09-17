"""
犀牛鸟项目测试
Rhino Bird Project Tests

测试核心功能
Tests for core functionality
"""

import sys
import os

# 添加src目录到Python路径
# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import RhinoBird, create_rhino_bird

def test_rhino_bird_creation():
    """测试犀牛鸟实例创建 / Test Rhino Bird instance creation"""
    bird = RhinoBird()
    assert bird.name == "犀牛鸟"
    assert bird.status == "ready"
    print("✅ 犀牛鸟创建测试通过 / Rhino Bird creation test passed")

def test_rhino_bird_fly():
    """测试犀牛鸟飞翔功能 / Test Rhino Bird flying functionality"""
    bird = RhinoBird("测试鸟")
    result = bird.fly()
    assert bird.status == "flying"
    assert "测试鸟" in result
    assert "飞翔" in result or "flying" in result
    print("✅ 犀牛鸟飞翔测试通过 / Rhino Bird flying test passed")

def test_rhino_bird_land():
    """测试犀牛鸟着陆功能 / Test Rhino Bird landing functionality"""
    bird = RhinoBird("测试鸟")
    bird.fly()  # 先飞翔 / First fly
    result = bird.land()
    assert bird.status == "landed"
    assert "测试鸟" in result
    assert "着陆" in result or "landed" in result
    print("✅ 犀牛鸟着陆测试通过 / Rhino Bird landing test passed")

def test_factory_function():
    """测试工厂函数 / Test factory function"""
    bird = create_rhino_bird("工厂鸟")
    assert bird.name == "工厂鸟"
    assert bird.status == "ready"
    print("✅ 工厂函数测试通过 / Factory function test passed")

def run_all_tests():
    """运行所有测试 / Run all tests"""
    print("🧪 开始运行犀牛鸟项目测试 / Starting Rhino Bird Project Tests")
    print("=" * 50)
    
    try:
        test_rhino_bird_creation()
        test_rhino_bird_fly()
        test_rhino_bird_land()
        test_factory_function()
        
        print("=" * 50)
        print("🎉 所有测试通过! / All tests passed!")
        return True
    except AssertionError as e:
        print(f"❌ 测试失败: {e} / Test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试错误: {e} / Test error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)