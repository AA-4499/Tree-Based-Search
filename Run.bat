@echo off
color 0A
title PathFinder AI - Route Finding Visualizer
set "BASE_DIR=%~dp0"
set "VENV_DIR=%BASE_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "VENV_READY=0"

:MENU
cls
echo ================================================
echo        PathFinder AI - Main Menu
echo ================================================
echo.
echo [1] Run Command Line Interface (CLI)
echo [2] Run Web GUI Visualizer
echo [3] Run CLI with specific algorithm
echo [4] Check Dependencies
echo [5] Install Dependencies
echo [6] View Help/Documentation
echo [7] Exit
echo.
echo ================================================
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto CLI
if "%choice%"=="2" goto WEBGUI
if "%choice%"=="3" goto CLI_ALGO
if "%choice%"=="4" goto CHECK_DEPS
if "%choice%"=="5" goto INSTALL_DEPS
if "%choice%"=="6" goto HELP
if "%choice%"=="7" goto EXIT
goto MENU

:CLI
cls
echo ================================================
echo          Command Line Interface (CLI)
echo ================================================
echo.
echo Format: python ^<file_name^> ^<method^>
echo Current file name: PathFinder-test.txt
echo Available Methods: BFS, DFS, GBFS, A*, CUS1, CUS2
echo.
set /p cmd="Enter command: python search.py PathFinder-test.txt "

if "%cmd%"=="" (
    echo Invalid input! Please enter a method name.
    pause
    goto CLI
)

call :ENSURE_VENV
if not "%VENV_READY%"=="1" goto MENU

cls
echo ================================================
echo Running Command: python search.py PathFinder-test.txt %cmd%
echo ================================================
echo.
call "%VENV_ACTIVATE%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)

python search.py PathFinder-test.txt %cmd%
set "RUN_EXIT_CODE=%ERRORLEVEL%"
call deactivate >nul
if not "%RUN_EXIT_CODE%"=="0" (
    echo.
    echo [INFO] Command exited with code %RUN_EXIT_CODE%.
)
echo.
echo ================================================
pause
goto MENU


:WEBGUI
cls
echo ================================================
echo        Starting Web GUI Visualizer
echo ================================================
echo.
call :ENSURE_VENV
if not "%VENV_READY%"=="1" goto MENU

call "%VENV_ACTIVATE%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)

echo Checking if Flask is installed...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Flask is not installed in the virtual environment!
    echo Please select option 5 to install dependencies.
    echo.
    call deactivate >nul
    pause
    goto MENU
)

echo Flask found!
echo Starting server...
echo.
echo ================================================
echo  Web GUI will open at: http://localhost:5000
echo  Press Ctrl+C to stop the server
echo ================================================
echo.
start http://localhost:5000
python search.py
set "RUN_EXIT_CODE=%ERRORLEVEL%"
call deactivate >nul
if not "%RUN_EXIT_CODE%"=="0" (
    echo.
    echo [INFO] Server exited with code %RUN_EXIT_CODE%.
    pause
)
goto MENU

:CLI_ALGO
cls
echo ================================================
echo     Run CLI with Specific Algorithm
echo ================================================
echo.
echo Available Algorithms:
echo [1] BFS  - Breadth-First Search
echo [2] DFS  - Depth-First Search
echo [3] GBFS - Greedy Best-First Search
echo [4] AS   - A* Search
echo [5] CUS1 - Depth-Limit Search
echo [6] CUS2 - Weighted A* Search
echo [0] Back to Main Menu
echo.
set /p algo_choice="Enter algorithm choice (0-6): "

if "%algo_choice%"=="0" goto MENU
if "%algo_choice%"=="1" set ALGO=BFS
if "%algo_choice%"=="2" set ALGO=DFS
if "%algo_choice%"=="3" set ALGO=GBFS
if "%algo_choice%"=="4" set ALGO=AS
if "%algo_choice%"=="5" set ALGO=CUS1
if "%algo_choice%"=="6" set ALGO=CUS2

if not defined ALGO (
    echo Invalid choice!
    pause
    goto CLI_ALGO
)

cls
echo ================================================
echo        Running %ALGO% Algorithm
echo ================================================
echo.
call :ENSURE_VENV
if not "%VENV_READY%"=="1" goto MENU

call "%VENV_ACTIVATE%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)

python search.py PathFinder-test.txt %ALGO%
set "RUN_EXIT_CODE=%ERRORLEVEL%"
call deactivate >nul
if not "%RUN_EXIT_CODE%"=="0" (
    echo.
    echo [INFO] Command exited with code %RUN_EXIT_CODE%.
)
echo.
echo ================================================
pause
goto MENU

:CHECK_DEPS
cls
echo ================================================
echo        Checking Dependencies
echo ================================================
echo.
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at "%VENV_DIR%".
    echo Please select option 5 to install dependencies.
    echo.
    pause
    goto MENU
)

echo Checking Python version in virtual environment...
"%VENV_PYTHON%" --version
echo.
echo Checking Flask (virtual environment)...
"%VENV_PYTHON%" -c "import flask; print('Flask version:', flask.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Flask is not installed in the virtual environment.
) else (
    echo [OK] Flask is installed in the virtual environment.
)
echo.
echo Checking required files...
if exist "search.py" (
    echo [OK] search.py found
) else (
    echo [ERROR] search.py not found!
)

if exist "PathFinder-test.txt" (
    echo [OK] PathFinder-test.txt found
) else (
    echo [ERROR] PathFinder-test.txt not found!
)

if exist "templates\index.html" (
    echo [OK] templates\index.html found
) else (
    echo [ERROR] templates\index.html not found!
)

echo.
echo ================================================
pause
goto MENU

:INSTALL_DEPS
cls
echo ================================================
echo        Installing Dependencies
echo ================================================
echo.
echo Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not found in PATH.
    echo Please install Python 3.x before continuing.
    echo.
    pause
    goto MENU
)

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment at "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python's venv module is available.
        echo.
        pause
        goto MENU
    )
)

call "%VENV_ACTIVATE%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    goto MENU
)

if exist "requirements.txt" (
    echo Installing from requirements.txt ...
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt
) else (
    echo requirements.txt not found. Installing Flask only...
    python -m pip install Flask
)

set "PIP_EXIT_CODE=%ERRORLEVEL%"
call deactivate >nul
if not "%PIP_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Failed to install one or more dependencies.
) else (
    echo.
    echo [SUCCESS] Dependencies installed successfully in the virtual environment!
)
echo.
echo ================================================
pause
goto MENU

:HELP
cls
echo ================================================
echo              Help / Documentation
echo ================================================
echo.
echo PROJECT: PathFinder AI - Route Finding Visualizer
echo.
echo DESCRIPTION:
echo   This project implements various tree-based search
echo   algorithms for solving route finding problems.
echo.
echo ALGORITHMS:
echo   BFS  - Breadth-First Search (Uninformed)
echo   DFS  - Depth-First Search (Uninformed)
echo   GBFS - Greedy Best-First Search (Informed)
echo   AS   - A* Search (Informed, Optimal)
echo   CUS1 - Depth-Limit Search (Uninformed)
echo   CUS2 - Weighted A* (Fast, suboptimal)
echo.
echo USAGE:
echo   CLI Mode:  python search.py [file] [algorithm]
echo   Web Mode:  python search.py (launches server for Web GUI)
echo.
echo INPUT FILE FORMAT:
echo   Nodes: id: (x,y)
echo   Edges: (from,to): weight
echo   Origin: node_id
echo   Destinations: id1;
id2; ...
echo.
echo REQUIREMENTS:
echo   - Python 3.x
echo   - Flask (for Web GUI only)
echo.
echo ================================================
pause
goto MENU

:EXIT
cls
echo ================================================
echo     Thank you for using PathFinder AI!
echo ================================================
echo.
timeout /t 2 >nul
exit

:ENSURE_VENV
if exist "%VENV_PYTHON%" (
    set "VENV_READY=1"
    goto :EOF
)
set "VENV_READY=0"
echo [ERROR] Virtual environment not found at "%VENV_DIR%".
echo Please run option 5 to install dependencies first.
pause
goto :EOF

:END
