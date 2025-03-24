@echo off
:START
echo Creating Virtual environment
python -m venv venv
echo Activating venv
call .\venv\Scripts\activate.bat
echo Installing dependencies
pip install openai rich gradio
goto :CHOICE

:CHOICE
echo.
echo ====================================================
echo Do you want to run the Gradio app with Gemma 3 4b it?
echo ====================================================
echo 1 - Yes
echo 0 - No

set /p whatapp=

if %whatapp%==1 (goto :RUNMODEL
) else if %whatapp%==0 (goto :QUIT 
) else (goto :INVALID)

:RUNMODEL
echo Starting Gradio application
python App_Gemma3-4b.py
goto :CHOICE

:INVALID
echo [44m [1m [93m
echo INVALID CHOICE
echo [0m
goto :CHOICE

:QUIT
echo [91mBYE BYE
echo [0m
pause
exit





PAUSE