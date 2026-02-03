@echo off
setlocal

REM 1) Set password for your Streamlit app
set "APP_PASSWORD=mmwwttpop

REM 2) Activate conda environment
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" new_phishing_detection

REM 3) Go to your project folder
cd /d "C:\Users\jjaz\Documents\FYP 2\Clean_Phishing_Detection_Model"

REM 4) Run Streamlit
streamlit run new_phishing_detector.py

endlocal
pause
