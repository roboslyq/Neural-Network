#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nginx日志分析启动器
这个脚本用于启动Nginx日志分析程序，并处理可能的依赖问题
"""

import os
import sys
import subprocess

def check_dependencies():
    """检查并安装必要的依赖项"""
    required_packages = [
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("正在安装缺少的依赖项...")
        for package in missing_packages:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("所有依赖项已安装完成！")

def main():
    """主函数"""
    print("="*50)
    print("Nginx 日志分析工具")
    print("="*50)
    print("本工具将分析数据分析目录下的所有.log文件，并生成可视化报告")
    print("分析结果将保存在 '数据分析/分析结果' 目录下")
    print("="*50)
    
    # 检查依赖项
    print("检查依赖项...")
    check_dependencies()
    
    # 导入日志分析模块
    print("启动分析程序...")
    from nginx_log_analysis import analyze_logs
    
    # 运行分析
    analyze_logs()
    
    print("\n分析完成！")
    print("="*50)
    print("可以在浏览器中打开 '数据分析/分析结果/report.html' 查看完整报告")
    print("="*50)

if __name__ == "__main__":
    main() 