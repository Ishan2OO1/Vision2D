@echo off
echo Building Object Recognition Project...
echo.

if not exist build mkdir build
cd build

cmake -G "Visual Studio 17 2022" -A x64 ..
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)

cd ..
echo.
echo Build completed successfully!
echo Executables are in bin\Release\
pause
