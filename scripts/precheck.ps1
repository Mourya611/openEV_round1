Param(
  [string]$EnvUrl = "http://localhost:7860"
)

$ErrorActionPreference = "Stop"

Write-Host "[CHECK] Required files"
$required = @(
  "app.py",
  "inference.py",
  "openenv.yaml",
  "Dockerfile",
  "README.md",
  "requirements.txt"
)
foreach ($file in $required) {
  if (-not (Test-Path $file)) {
    throw "Missing required file: $file"
  }
}

Write-Host "[CHECK] HTTP health"
$health = Invoke-RestMethod -Method Get -Uri "$EnvUrl/health"
if (-not $health.ok) {
  throw "Health check failed"
}

Write-Host "[CHECK] Tasks"
$tasks = Invoke-RestMethod -Method Get -Uri "$EnvUrl/tasks"
$taskCount = @($tasks.tasks.PSObject.Properties.Name).Count
if ($taskCount -lt 3) {
  throw "At least 3 tasks are required"
}

Write-Host "[CHECK] reset + state"
$null = Invoke-RestMethod -Method Post -Uri "$EnvUrl/reset" -ContentType "application/json" -Body '{"task_name":"ticket-triage-easy"}'
$state = Invoke-RestMethod -Method Get -Uri "$EnvUrl/state"
if (-not $state.task_name) {
  throw "state() did not return task_name"
}

Write-Host "[OK] Precheck passed."
