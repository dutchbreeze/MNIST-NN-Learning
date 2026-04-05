@echo off
echo Creating TrainingData subfolders with .gitkeep files...
for /L %%i in (0,1,9) do (
    if not exist "TrainingData\%%i" mkdir "TrainingData\%%i"
    echo. > "TrainingData\%%i\.gitkeep"
)
echo Done! Place your 28x28 images in TrainingData\0 through TrainingData\9.
pause
