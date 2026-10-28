@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   BIEN DICH REPORT_P4AIDS (main.tex -^> main.pdf)
echo ============================================

if exist "bin\tectonic.exe" (
    bin\tectonic.exe main.tex
) else (
    tectonic main.tex
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [THANH CONG] File main.pdf da duoc tao thanh cong!
) else (
    echo.
    echo [THAT BAI] Co loi trong qua trinh bien dich.
)
endlocal
pause
