# AI-SC v2 campaign supervisor.
# Resumes the campaign (the driver is lock-guarded + resume-safe) and, once all
# 30 runs are complete, builds the paper. Safe to run repeatedly: it exits early
# if a campaign process is already running. Triggered at logon + periodically so
# the campaign survives the PC's reboots while the user is away.
$ErrorActionPreference = "Continue"
$code    = "C:\Users\luisl\repos\AI-SC\nod\code"
$results = "C:\Users\luisl\repos\AI-SC\nod\results"
$py      = "C:\Users\luisl\anaconda3\envs\jax-env-3.11\python.exe"
$env:OLLAMA_MODEL = "gemma3:12b"     # v3 re-run: e4b failed JSON (~85% planner / 100% reviewer fallback); 12b is reliable
$env:OLLAMA_NUM_CTX = "8192"         # default 2048 truncated long prompts -> empty JSON
$slog = Join-Path $results "supervisor.log"
function Log($m) { "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -FilePath $slog -Append -Encoding utf8 }

# Guard: do not start a second runner.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -like '*run_campaign*' }
if ($running) { Log "campaign already running (PID $($running.ProcessId)); exit."; exit 0 }

# Ensure ollama is up (LLM jobs need it; the driver also waits, but start it proactively).
if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    $ol = "C:\Users\luisl\AppData\Local\Programs\Ollama\ollama.exe"
    if (Test-Path $ol) { Start-Process -FilePath $ol -ArgumentList "serve" -WindowStyle Hidden; Log "started ollama serve"; Start-Sleep -Seconds 20 }
}

Log "supervisor: launching campaign driver"
& $py -u (Join-Path $code "run_campaign.py") *>> (Join-Path $results "campaign.log")
Log "campaign driver returned (rc=$LASTEXITCODE)"

# Build the paper once every run has a FINAL.json.
$swarm = Join-Path $results "swarm_runs"
$done = 0
if (Test-Path $swarm) {
    $done = (Get-ChildItem $swarm -Directory -Filter "v2_*" -ErrorAction SilentlyContinue |
             Where-Object { Test-Path (Join-Path $_.FullName "FINAL.json") }).Count
}
Log "completed v2 runs: $done / 30"
if ($done -ge 30) {
    $sentinel = Join-Path $results "PAPER_BUILT"
    if (-not (Test-Path $sentinel)) {
        Log "all runs complete -> building paper"
        & $py -u (Join-Path $code "build_paper.py") *>> (Join-Path $results "build_paper.log")
        if ($LASTEXITCODE -eq 0) {
            Set-Content $sentinel (Get-Date).ToString() ; Log "paper built OK"
            # Reboot-proof, on-PC notification that the campaign + paper are done.
            $paper = Resolve-Path (Join-Path $results "..\paper\main.pdf") -ErrorAction SilentlyContinue
            "AI-SC campaign COMPLETE $([DateTime]::Now). Paper: $paper" |
                Set-Content (Join-Path ([Environment]::GetFolderPath('Desktop')) "AISC_CAMPAIGN_DONE.txt")
            if ($paper) { try { Start-Process $paper.Path } catch {} }
            try { & msg.exe * "AI-SC: campaign finished. The paper (main.pdf) is ready (see AISC_CAMPAIGN_DONE.txt on your Desktop)." } catch {}
        }
        else { Log "build_paper failed rc=$LASTEXITCODE" }
    } else { Log "paper already built (sentinel present)" }
}
Log "supervisor done"
