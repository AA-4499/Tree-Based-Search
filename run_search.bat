@echo off
color 0A
title PathFinder AI - Route Finding Visualizer

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

cls
echo ================================================
echo Running Command: python search.py PathFinder-test.txt %cmd%
echo ================================================
echo.
python search.py PathFinder-test.txt %cmd%
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
echo Checking if Flask is installed...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Flask is not installed!
    echo Please select option 5 to install dependencies.
    echo.
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
echo [5] CUS1 - Custom Uninformed Search
echo [6] CUS2 - Weighted A* Search (w=10)
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
python search.py PathFinder-test.txt %ALGO%
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
echo Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python from https://www.python.org/
) else (
    echo [OK] Python is installed
)
echo.
echo Checking Flask...
python -c "import flask; print('Flask version:', flask.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Flask is not installed
) else (
    echo [OK] Flask is installed
)
echo.
echo Checking required files...
if exist "search.py" (
    echo [OK] search.py found
) else (
    echo [ERROR] search.py not found!
)

if exist "app.py" (
    echo [OK] app.py found
) else (
    echo [ERROR] app.py not found!
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
echo Installing Flask...
pip install flask
echo.
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    echo Make sure pip is installed and working.
) else (
    echo [SUCCESS] Dependencies installed successfully!
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
echo   CUS1 - Custom Uninformed (First path found)
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

:END