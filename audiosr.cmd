@echo off
rem audiosr.cmd - Windows entry point for the audiosr script.
rem
rem Prefers a venv next to this file (venv\Scripts\python.exe), then whatever
rem python is on PATH. The audio backend defaults to portaudio on Windows.
rem
rem Usage: audiosr in.mp3 out.flac
rem        audiosr --check
rem        audiosr --list-devices
setlocal
set "HERE=%~dp0"
set "SCRIPT=%HERE%audiosr"
if exist "%HERE%venv\Scripts\python.exe" (
  "%HERE%venv\Scripts\python.exe" "%SCRIPT%" %*
) else (
  python "%SCRIPT%" %*
)
exit /b %ERRORLEVEL%
