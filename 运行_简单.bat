@echo off
chcp 65001 >nul
echo ============================================
echo SEMixer 复现 - 简单数据集
echo ============================================
echo.
echo 数据集：ETTh1, ETTh2
echo 训练次数：2 x 4 x 5 = 40 次
echo 预计时间：约 30 分钟
echo.
echo ============================================

cd /d %~dp0
call .venv\Scripts\activate

echo 开始运行...
python run_reproduction.py easy --collect-after

echo.
echo ============================================
echo 简单数据集完成!
echo ============================================
pause
