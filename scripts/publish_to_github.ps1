param(
    [string]$UserName = "keys2023190905023",
    [string]$RepoName = "pynq-z2-ps-pl-detector-scheme",
    [string]$Token = $env:GITHUB_TOKEN,
    [switch]$Private,
    [switch]$SkipCreate
)

$ErrorActionPreference = "Stop"

if (-not $Token) {
    throw "Missing GitHub token. Pass -Token <PAT> or set GITHUB_TOKEN."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$visibility = if ($Private) { $true } else { $false }
$remoteUrl = "https://github.com/$UserName/$RepoName.git"
$apiUrl = "https://api.github.com/user/repos"

$headers = @{
    Authorization = "Bearer $Token"
    Accept        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent"  = "codex-publisher"
}

$body = @{
    name         = $RepoName
    private      = $visibility
    has_issues   = $true
    has_projects = $false
    has_wiki     = $false
    auto_init    = $false
} | ConvertTo-Json

if (-not $SkipCreate) {
    try {
        Invoke-RestMethod -Method Post -Uri $apiUrl -Headers $headers -Body $body -ContentType "application/json" | Out-Null
    } catch {
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode.value__ -ne 422) {
            throw
        }
    }
}

git -C $repoRoot remote remove origin 2>$null
git -C $repoRoot remote add origin $remoteUrl

$credentialUrl = "https://$UserName`:$Token@github.com/$UserName/$RepoName.git"
git -C $repoRoot push $credentialUrl main:main
if ($LASTEXITCODE -ne 0) {
    throw "git push failed with exit code $LASTEXITCODE"
}

Write-Host "PUBLISH_OK"
Write-Host $remoteUrl
