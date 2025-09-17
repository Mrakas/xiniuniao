"""
犀牛鸟项目核心模块
Rhino Bird Project Core Module

这个模块包含项目的核心功能
This module contains the core functionality of the project
"""

class RhinoBird:
    """犀牛鸟核心类 / Rhino Bird Core Class"""
    
    def __init__(self, name="犀牛鸟"):
        """
        初始化犀牛鸟实例
        Initialize Rhino Bird instance
        
        Args:
            name (str): 实例名称 / Instance name
        """
        self.name = name
        self.status = "ready"
    
    def fly(self):
        """
        让犀牛鸟飞翔
        Make the Rhino Bird fly
        
        Returns:
            str: 飞翔状态消息 / Flying status message
        """
        self.status = "flying"
        return f"{self.name} 正在飞翔! / {self.name} is flying!"
    
    def land(self):
        """
        让犀牛鸟着陆
        Make the Rhino Bird land
        
        Returns:
            str: 着陆状态消息 / Landing status message
        """
        self.status = "landed"
        return f"{self.name} 已经着陆! / {self.name} has landed!"
    
    def get_status(self):
        """
        获取当前状态
        Get current status
        
        Returns:
            str: 当前状态 / Current status
        """
        return self.status

def create_rhino_bird(name="犀牛鸟"):
    """
    创建犀牛鸟实例的工厂函数
    Factory function to create Rhino Bird instance
    
    Args:
        name (str): 犀牛鸟名称 / Rhino Bird name
        
    Returns:
        RhinoBird: 犀牛鸟实例 / Rhino Bird instance
    """
    return RhinoBird(name)