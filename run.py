"""
基于轴承振动信号的智能故障诊断系统
启动入口文件
"""
from app import create_app
from config import Config

app = create_app(Config)

if __name__ == '__main__':
    print("=" * 60)
    print("  基于轴承振动信号的智能故障诊断系统")
    print("  Intelligent Bearing Fault Diagnosis System")
    print("=" * 60)
    print(f"  访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
