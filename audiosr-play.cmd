@echo off
rem audiosr-play.cmd - play an audio file through FlashSR (Windows).
rem
rem Usage: audiosr-play track.mp3 [--window 2.56] [--output-device NAME]
rem        audiosr-play track.mp3 --dump out.flac
rem        audiosr-play --list-devices
call "%~dp0audiosr.cmd" --play %*
exit /b %ERRORLEVEL%
