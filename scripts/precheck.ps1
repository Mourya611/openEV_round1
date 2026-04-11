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

Write-Host "[CHECK] score bounds across all tasks"
$taskNames = @($tasks.tasks.PSObject.Properties.Name)
foreach ($taskName in $taskNames) {
  $null = Invoke-RestMethod -Method Post -Uri "$EnvUrl/reset" -ContentType "application/json" -Body (@{ task_name = $taskName } | ConvertTo-Json)
  $state = Invoke-RestMethod -Method Get -Uri "$EnvUrl/state"

  while (-not $state.done) {
    $ticket = $state.tickets[$state.current_index]
    $action = switch ($ticket.ticket_id) {
      "E-1001" { @{ ticket_id="E-1001"; decision="resolve"; priority="medium"; response_template="empathetic"; notes="Issue refund for duplicate invoice and confirm corrected billing." } }
      "E-1002" { @{ ticket_id="E-1002"; decision="escalate"; priority="urgent"; response_template="compliance"; notes="Escalate security incident and begin containment review immediately." } }
      "M-2001" { @{ ticket_id="M-2001"; decision="escalate"; priority="high"; response_template="technical"; notes="Escalate to reproduce the browser export bug in Chrome." } }
      "M-2002" { @{ ticket_id="M-2002"; decision="request_info"; priority="medium"; response_template="technical"; notes="Request logs and timestamps for the recurring integration disconnect." } }
      "M-2003" { @{ ticket_id="M-2003"; decision="escalate"; priority="high"; response_template="compliance"; notes="Escalate for legal review of the DPA wording request." } }
      "H-3001" { @{ ticket_id="H-3001"; decision="escalate"; priority="urgent"; response_template="compliance"; notes="Escalate security incident and start containment for unusual key usage." } }
      "H-3002" { @{ ticket_id="H-3002"; decision="request_info"; priority="high"; response_template="empathetic"; notes="Request contract details to verify the annual discount issue." } }
      "H-3003" { @{ ticket_id="H-3003"; decision="escalate"; priority="medium"; response_template="technical"; notes="Escalate webhook delay issue and validate idempotency handling." } }
      "H-3004" { @{ ticket_id="H-3004"; decision="escalate"; priority="urgent"; response_template="compliance"; notes="Escalate SOC2 and subprocessor evidence request to compliance." } }
      default { throw "No precheck action defined for ticket $($ticket.ticket_id)" }
    }

    $step = Invoke-RestMethod -Method Post -Uri "$EnvUrl/step" -ContentType "application/json" -Body ($action | ConvertTo-Json)
    if (($step.reward -le 0.0) -or ($step.reward -ge 1.0)) {
      throw "Reward out of open interval for task ${taskName}: $($step.reward)"
    }

    if ($step.done) {
      $finalScore = $step.info.final_score
      if (($null -eq $finalScore) -or ($finalScore -le 0.0) -or ($finalScore -ge 1.0)) {
        throw "Final score out of open interval for task ${taskName}: $finalScore"
      }
      break
    }

    $state = Invoke-RestMethod -Method Get -Uri "$EnvUrl/state"
  }
}

Write-Host "[OK] Precheck passed."
