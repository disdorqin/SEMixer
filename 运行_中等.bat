@echo off
chcp 65001 >nul
echo ============================================
echo SEMixer 复现 - 中等难度数据集
echo ============================================
echo.
echo 数据集：ETTm1, ETTm2, exchange_rate, national_illness
echo 训练次数：4 x 4 x 5 = 80 次
echo 预计时间：约 2 小时
echo.
echo ============================================

cd /d %~dp0
call .venv\Scripts\activate

echo 开始运行...
python run_reproduction.py medium --collect-after

echo.
echo ============================================
echo 中等难度数据集完成!
echo ============================================
pause