@echo off
rem audiosr-sink.cmd - middleware mode on Windows.
rem
rem Windows has no user-space virtual sink, so this needs a virtual cable
rem (VB-CABLE, VoiceMeeter). Point your apps at the cable and capture its other
rem end:
rem
rem   audiosr-sink --input-device "CABLE Output" --output-device "Speakers"
call "%~dp0audiosr.cmd" --listen %*
exit /b %ERRORLEVEL%
