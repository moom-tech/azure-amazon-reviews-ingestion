# Azure CLI login and data upload script
Write-Host "Checking Azure CLI authentication..." -ForegroundColor Green

# Try to show current account (will fail if not authenticated)
$account = az account show 2>$null

if ($null -eq $account) {
    Write-Host "Not authenticated. Logging in..." -ForegroundColor Yellow
    az login --use-device-code
}

# Upload the dataset
Write-Host "Uploading dataset to Azure..." -ForegroundColor Green
az storage blob upload-batch `
  --account-name lab5amlworkspa4258181091 `
  --destination cmapss-data `
  --source ./data/raw/ `
  --pattern "*.txt"

Write-Host "Upload complete!" -ForegroundColor Green
