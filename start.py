#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键启动脚本
"""
import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("  轴承智能故障诊断系统 - 一键启动")
    print("=" * 50)
    
    # 检查依赖
    print("\n[1/2] 检查依赖...")
    try:
        import flask
        import numpy
        import sklearn
        print("  ✓ 依赖已安装")
    except ImportError:
        print("  正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 启动服务
    print("\n[2/2] 启动B/S后端服务...")
    print("  访问地址: http://127.0.0.1:5000")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)
    
    os.system(f"{sys.executable} run.py")

if __name__ == "__main__":
    main()
