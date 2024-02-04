@echo off
setlocal

:: set Intel environment: MKL
set INTEL_ENV="C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if exist %INTEL_ENV% (
    call %INTEL_ENV% intel64 vs2022
) else (
	echo Error: Intel MKL not found
	exit 1
)

:: check CMake existing
set CMAKE="C:\Program Files\CMake\bin\cmake.exe"
if not exist %CMAKE% (
    echo Error: CMake not found
	exit 2
)

:: set environment variables
set BOOST_ROOT=C:\LIBS64\Boost\boost_1_82_0
set CASES_PATH=C:/WorkDirectory/Cases
set AMGX_DIR=C:/LIBS64/AMGX

%CMAKE% -Ax64 -B build ^
	-DPOLYSOLVER_TEST=OFF ^
	-DPOLYSOLVER_USE_CUDA=ON ^
	-DMKL_ARCH=intel64 ^
	-DMKL_LINK=static ^
	-DMKL_INTERFACE=lp64

pause
endlocal