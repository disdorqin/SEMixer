@echo off
chcp 65001 >nul
echo ============================================
echo SEMixer 复现 - 困难数据集
echo ============================================
echo.
echo 数据集：weather, electricity, traffic, solar_AL
echo 训练次数：4 x 4 x 5 = 80 次
echo 预计时间：约 8 小时（需要好显卡）
echo.
echo 警告：这些数据集变量多，耗时长，耗显存
echo ============================================

cd /d %~dp0repo
call ..\.venv\Scripts\activate

echo 开始运行...
python run.py hard

echo.
echo ============================================
echo 困难数据集完成!
echo 运行 collect_results.py 查看结果
echo ============================================
pause