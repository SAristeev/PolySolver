@echo ON
setlocal

:: set Intel environment: MKL
set INTEL_ENV="C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if exist %INTEL_ENV% (
    call %INTEL_ENV%" intel64 vs2022
) else (
	echo Error: Intel MKL not found
)

:: check CMake existing
set CMAKE="C:\Program Files\CMake\bin\cmake.exe"
if exist %CMAKE% (
    call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022
) else (
	echo Error: CMake not found
)

:: set environment variables
set BOOST_ROOT=C:\LIBS64\Boost\boost_1_82_0
set CASES_PATH=C:/WorkDirectory/Cases
set AMGX_DIR=C:/LIBS64/AMGX

%CMAKE% -Ax64 -B build ^
	-DPOLYSOLVER_TEST=OFF ^
	-DPOLYSOLVER_USE_CUDA=OFF ^
	-DPOLYSOLVER_USE_AMGCL=ON ^
	-DPOLYSOLVER_USE_MKL=ON

pause
endlocal