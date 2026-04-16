@echo off
chcp 65001 >nul
echo ============================================
echo SEMixer 论文复现脚本
echo ============================================
echo.

cd /d %~dp0
call .venv\Scripts\activate

echo 步骤 1: 验证环境...
python run_reproduction.py check

echo.
echo 步骤 2: 启动 Python 复现入口...
python run_reproduction.py easy --collect-after

echo.
echo ============================================
echo 复现脚本执行完毕
echo ============================================
pause