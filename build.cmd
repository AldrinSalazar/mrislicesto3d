@echo off
REM Build script for mrislicesto3d

REM Clean up previous builds
if exist mrislicesto3d del mrislicesto3d
if exist mrislicesto3d_mac_arm64 del mrislicesto3d_mac_arm64
if exist mrislicesto3d_linux_amd64 del mrislicesto3d_linux_amd64
if exist mrislicesto3d_windows_amd64.exe del mrislicesto3d_windows_amd64.exe

REM Run tests
go test .\...

REM Build for macOS (commented out as this would typically be done on a Mac)
REM set CGO_ENABLED=1
REM set GOARCH=arm64
REM set GOOS=darwin
REM go build -o mrislicesto3d_mac_arm64 -ldflags="-s -w" .\cmd\mrislicesto3d\main.go

REM Build for Linux (commented out as this would typically be done on Linux)
REM set CGO_ENABLED=1
REM set GOARCH=amd64
REM set GOOS=linux
REM go build -o mrislicesto3d_linux_amd64 -ldflags="-s -w" .\cmd\mrislicesto3d\main.go

REM Build for Windows
set CGO_ENABLED=1
set GOARCH=amd64
set GOOS=windows
go build -o mrislicesto3d_windows_amd64.exe -ldflags="-s -w" .\cmd\mrislicesto3d\main.go
