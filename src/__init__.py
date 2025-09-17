"""
犀牛鸟项目 (Rhino Bird Project)
主要模块初始化文件

Main module initialization file for the Rhino Bird Project
"""

__version__ = "0.1.0"
__author__ = "Rhino Bird Team"
__description__ = "犀牛鸟项目 - 高效智能的开源解决方案"

def get_version():
    """获取项目版本 / Get project version"""
    return __version__

def get_info():
    """获取项目信息 / Get project information"""
    return {
        "name": "犀牛鸟项目 (Rhino Bird Project)",
        "version": __version__,
        "author": __author__,
        "description": __description__
    }