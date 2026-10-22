$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn web_demo.app:app --host 127.0.0.1 --port 8000 --no-access-log
