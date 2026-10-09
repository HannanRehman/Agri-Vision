# One-time deletion of individually verified ZIP duplicates, authorized by user.
# Does not recursively delete a tree and never touches merged training data.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$expectedChiliSource = [System.IO.Path]::GetFullPath('F:\Agri Vision\archive\imported_datasets\PENYAKIT PADA CABAI.v1i.yolov8')
$resolvedChiliSource = (Resolve-Path -LiteralPath $expectedChiliSource).Path
if (-not $resolvedChiliSource.Equals($expectedChiliSource,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected resolved source' }
$chiliSourcePrefix = $resolvedChiliSource + '\'
$chiliZipPath = $expectedChiliSource + '.zip'
$chiliSourceFiles = @(Get-ChildItem -LiteralPath $resolvedChiliSource -File -Recurse)
$verifiedSourceFiles = [System.Collections.Generic.List[object]]::new()
$chiliZip = [System.IO.Compression.ZipFile]::OpenRead($chiliZipPath)
try {
    $zipFileEntries = @($chiliZip.Entries | Where-Object { $_.Name -ne '' })
    if ($chiliSourceFiles.Count -ne $zipFileEntries.Count) { throw 'ZIP and source file counts differ' }
    foreach ($sourceFile in $chiliSourceFiles) {
        if (-not $sourceFile.FullName.StartsWith($chiliSourcePrefix,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'File outside intended source directory' }
        if (($sourceFile.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Refusing a reparse-point file' }
        $entryName = $sourceFile.FullName.Substring($chiliSourcePrefix.Length).Replace('\','/')
        $zipEntry = $chiliZip.GetEntry($entryName)
        if ($null -eq $zipEntry -or $zipEntry.Length -ne $sourceFile.Length) { throw "ZIP entry mismatch: $entryName" }
        $entryStream = $zipEntry.Open()
        $digestAlgorithm = [System.Security.Cryptography.SHA256]::Create()
        try { $entryHash = [System.BitConverter]::ToString($digestAlgorithm.ComputeHash($entryStream)).Replace('-','') }
        finally { $entryStream.Dispose(); $digestAlgorithm.Dispose() }
        $sourceHash = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
        if ($sourceHash -ne $entryHash) { throw "ZIP bytes differ: $entryName" }
        $verifiedSourceFiles.Add([PSCustomObject]@{path=$sourceFile.FullName; relative=$entryName; sha256=$sourceHash; bytes=$sourceFile.Length})
    }
} finally { $chiliZip.Dispose() }

# Delete only named, verified duplicates. Every removal has an intact ZIP copy.
foreach ($sourceFile in $verifiedSourceFiles) { Remove-Item -LiteralPath $sourceFile.path }
$emptyChiliDirectories = @(Get-ChildItem -LiteralPath $resolvedChiliSource -Directory -Recurse | Sort-Object { $_.FullName.Length } -Descending)
foreach ($emptyDirectory in $emptyChiliDirectories) {
    if (-not $emptyDirectory.FullName.StartsWith($chiliSourcePrefix,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Directory outside intended source' }
    if (@(Get-ChildItem -LiteralPath $emptyDirectory.FullName -Force).Count -ne 0) { throw 'Directory is not empty' }
    Remove-Item -LiteralPath $emptyDirectory.FullName
}
if (@(Get-ChildItem -LiteralPath $resolvedChiliSource -Force).Count -ne 0) { throw 'Source directory is not empty' }
Remove-Item -LiteralPath $resolvedChiliSource
$verifiedSourceFiles | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'F:\Agri Vision\reports\chili_source_deleted_files_20261009.json' -Encoding utf8
Write-Output ('Removed ' + $verifiedSourceFiles.Count + ' verified source duplicates; compressed ZIP retained.')
