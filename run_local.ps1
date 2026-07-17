# -----------------------------------------------------------------------------
# RapRank - Local Startup Script (Run without Docker)
# -----------------------------------------------------------------------------
# This script starts all necessary microservices, databases, and frontends
# in separate, persistent PowerShell windows for local development.
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Helper function to check if a port is in use
function Test-PortInUse($port) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

# Define services, their ports, and their start commands
$services = @(
    @{
        Name = "PostgreSQL"
        Port = 5432
        Required = $true
        Action = {
            Write-Host "[-] PostgreSQL must be running on port 5432 (database: raprank_db, user: raprank_user)." -ForegroundColor Yellow
            Write-Host "    Please ensure your local PostgreSQL server service is started." -ForegroundColor Yellow
        }
    },
    @{
        Name = "Redis"
        Port = 6379
        Required = $true
        Action = {
            Write-Host "[*] Checking for Redis executable..." -ForegroundColor Cyan
            $redisPath = (Get-Command redis-server -ErrorAction SilentlyContinue).Source
            if (-not $redisPath) {
                # Fallback to the Winget-installed path
                $fallbackPath = "C:\Users\Savyasachi Mishra\AppData\Local\Microsoft\WinGet\Packages\taizod1024.redis-windows-fork_Microsoft.Winget.Source_8wekyb3d8bbwe\Redis-8.8.0-Windows-x64-msys2\redis-server.exe"
                if (Test-Path $fallbackPath) {
                    $redisPath = $fallbackPath
                }
            }

            if ($redisPath) {
                Write-Host "[+] Found Redis at: $redisPath" -ForegroundColor Green
                Write-Host "[+] Starting Redis in a new window..." -ForegroundColor Green
                Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - Redis'; Write-Host 'Starting Redis Server...' -ForegroundColor Cyan; & '$redisPath'"
            } else {
                Write-Host "[-] Redis is not running and redis-server.exe was not found in PATH." -ForegroundColor Red
                Write-Host "    Please start the Redis service manually on port 6379." -ForegroundColor Red
            }
        }
    },
    @{
        Name = "Hindi Semantic Scoring (7860)"
        Port = 7860
        Required = $false
        Action = {
            Write-Host "[+] Starting Hindi Semantic Scoring Service on port 7860..." -ForegroundColor Green
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - Semantic Service'; cd '$PSScriptRoot\raprank-semantic'; Write-Host 'Starting Hindi Semantic scoring service...' -ForegroundColor Cyan; .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 7860"
        }
    },
    @{
        Name = "NLP Scoring (8000)"
        Port = 8000
        Required = $false
        Action = {
            Write-Host "[+] Starting NLP Scoring Service on port 8000..." -ForegroundColor Green
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - NLP Service'; cd '$PSScriptRoot\raprank-nlp'; Write-Host 'Starting NLP scoring service...' -ForegroundColor Cyan; .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000"
        }
    },
    @{
        Name = "Audio Upload (9090)"
        Port = 9090
        Required = $false
        Action = {
            Write-Host "[+] Starting Go Audio Upload Service on port 9090..." -ForegroundColor Green
            # Ensure upload directories exist
            New-Item -ItemType Directory -Force -Path "$PSScriptRoot\raprank-upload\uploads\audio" | Out-Null
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - Audio Upload'; cd '$PSScriptRoot\raprank-upload'; Write-Host 'Starting Go Upload service...' -ForegroundColor Cyan; go run main.go"
        }
    },
    @{
        Name = "Spring Boot Backend (8080)"
        Port = 8080
        Required = $false
        Action = {
            Write-Host "[+] Starting Spring Boot Backend on port 8080..." -ForegroundColor Green
            New-Item -ItemType Directory -Force -Path "$PSScriptRoot\raprank-backend\uploads\audio" | Out-Null
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - Spring Boot Backend'; cd '$PSScriptRoot\raprank-backend'; Write-Host 'Starting Spring Boot backend...' -ForegroundColor Cyan; mvn spring-boot:run"
        }
    },
    @{
        Name = "Vite Frontend (3000)"
        Port = 3000
        Required = $false
        Action = {
            Write-Host "[+] Starting Vite Frontend on port 3000..." -ForegroundColor Green
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'RapRank - Vite Frontend'; cd '$PSScriptRoot'; Write-Host 'Starting Vite frontend...' -ForegroundColor Cyan; npm run dev"
        }
    }
)

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "             RapRank Bare-Metal Launcher                  " -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Magenta

# Check and start services
foreach ($service in $services) {
    $name = $service.Name
    $port = $service.Port
    
    if (Test-PortInUse $port) {
        Write-Host "[OK] $name is already running on port $port." -ForegroundColor Gray
    } else {
        & $service.Action
    }
}

Write-Host "`n==========================================================" -ForegroundColor Magenta
Write-Host "Launcher completed. Check spawned windows for service status." -ForegroundColor Magenta
Write-Host "Frontend will be available at: http://localhost:3000" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Magenta
