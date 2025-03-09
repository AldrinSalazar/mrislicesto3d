#!/bin/bash

# Build for macOS
rm -f mrislicesto3d
rm -f mrislicesto3d_mac_arm64
rm -f mrislicesto3d_linux_amd64
rm -f mrislicesto3d_windows_amd64.exe

go test ./...
CGO_ENABLED=1 GOARCH=arm64 GOOS=darwin go build -o mrislicesto3d_mac_arm64 -ldflags="-s -w" ./cmd/mrislicesto3d/main.go

# Build for Linux
#CGO_ENABLED=1 GOARCH=amd64 GOOS=linux go build -o mrislicesto3d_linux_amd64 -ldflags="-s -w" ./cmd/mrislicesto3d/main.go

# Build for Windows
#CGO_ENABLED=1 GOARCH=amd64 GOOS=windows go build -o mrislicesto3d_windows_amd64.exe -ldflags="-s -w" ./cmd/mrislicesto3d/main.go
