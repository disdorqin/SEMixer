@echo off
chcp 65001 >nul
echo ============================================
echo SEMixer 复现 - 全部数据集
echo ============================================
echo.
echo 数据集：10 个全部
echo 训练次数：10 x 4 x 5 = 200 次
echo 预计时间：约 10-15 小时
echo.
echo 警告：这将运行所有数据集，耗时较长
echo 建议夜间运行，早上查看结果
echo ============================================

cd /d %~dp0
call .venv\Scripts\activate

echo 开始运行...
python run_reproduction.py all --collect-after

echo.
echo ============================================
echo 全部数据集完成!
echo ============================================
pause