@echo off
REM Starts the whole stack and prints the startup banner with working
REM links in the console (docker compose up -d is detached, so it does
REM not display anything by itself).
REM
REM Usage (PowerShell or cmd):  .\up.cmd   (or double-click in Explorer)

docker compose up -d

REM Foreground banner: waits for every service to answer, prints links,
REM then removes itself (--rm).
docker compose run --rm banner
