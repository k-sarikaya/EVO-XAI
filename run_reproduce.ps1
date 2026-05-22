$ErrorActionPreference = "Stop"

Write-Host "[info] Running from: $PSScriptRoot"

# Prefer the packaged scripts in ./code
$python = "python"

Write-Host "[info] Python version:"
& $python -c "import sys; print(sys.version)"

Write-Host "[step] Validate snapshot integrity"
& $python ".\\validation\\validate_package.py"

Write-Host "[step] (Optional) Reproduce core artifacts (may be slow)"
Write-Host "       If you want to run it, uncomment the next line."
# & $python ".\\code\\reproduce_hiss_submission.py"

Write-Host "[ok] Done"

