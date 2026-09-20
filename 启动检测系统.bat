@echo off
chcp 65001 >nul
echo ========================================
echo   轴承故障诊断系统 - 启动中...
echo ========================================
echo.
cd /d "C:\Users\LENOVO\Doubao\chats\2026-09-11\new-chat\iot-work"
start "" http://127.0.0.1:5000
python run.py
pause
