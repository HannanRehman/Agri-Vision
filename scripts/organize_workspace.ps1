# One-time, reviewed local organization. Never deletes datasets or checkpoints.
$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$workspacePrefix = $workspaceRoot + '\'
$cleanupLog = [System.Collections.Generic.List[object]]::new()

function Resolve-WorkspaceTarget([string]$relative) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
    if (-not $target.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Target is outside the workspace: $target"
    }
    return $target
}

function Archive-WorkspaceItem([string]$sourceRelative, [string]$destinationRelative) {
    $source = Resolve-WorkspaceTarget $sourceRelative
    $destination = Resolve-WorkspaceTarget $destinationRelative
    if (-not (Test-Path -LiteralPath $source)) { throw "Source missing: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    $resolvedSource = (Resolve-Path -LiteralPath $source).Path
    if (-not $resolvedSource.StartsWith($workspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved source is outside the workspace: $resolvedSource"
    }
    $item = Get-Item -LiteralPath $source -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Refusing to move a reparse point' }
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Move-Item -LiteralPath $resolvedSource -Destination $destination
    $cleanupLog.Add([PSCustomObject]@{action='move'; source=$sourceRelative; destination=$destinationRelative})
}

function Remove-IdenticalFile([string]$duplicateRelative, [string]$keepRelative) {
    $duplicate = Resolve-WorkspaceTarget $duplicateRelative
    $keep = Resolve-WorkspaceTarget $keepRelative
    $duplicateHash = (Get-FileHash -LiteralPath $duplicate -Algorithm SHA256).Hash
    $keepHash = (Get-FileHash -LiteralPath $keep -Algorithm SHA256).Hash
    if ($duplicateHash -ne $keepHash) { throw "Files differ: $duplicateRelative" }
    $bytes = (Get-Item -LiteralPath $duplicate).Length
    Remove-Item -LiteralPath $duplicate
    $cleanupLog.Add([PSCustomObject]@{action='remove duplicate file'; removed=$duplicateRelative; retained=$keepRelative; sha256=$keepHash; bytes=$bytes})
}

try {
    $audit = Get-Content -LiteralPath (Resolve-WorkspaceTarget 'reports/workspace_audit_before_20261009.json') -Raw | ConvertFrom-Json
    $comparison = $audit.round_035_evaluation_folder_comparison
    if (-not $comparison.identical) { throw 'Evaluation folder fingerprints differ' }
    $duplicateFolder = Resolve-WorkspaceTarget 'saved_models/test_evaluation_round_035-2'
    $keepFolder = Resolve-WorkspaceTarget 'saved_models/test_evaluation_round_035'
    $duplicateFiles = @(Get-ChildItem -LiteralPath $duplicateFolder -File -Recurse)
    $keepFiles = @(Get-ChildItem -LiteralPath $keepFolder -File -Recurse)
    if ($duplicateFiles.Count -ne $keepFiles.Count -or $duplicateFiles.Count -ne 6) { throw 'Unexpected evaluation folder contents' }
    foreach ($duplicate in $duplicateFiles) {
        $relative = $duplicate.FullName.Substring($duplicateFolder.Length + 1)
        $keep = Join-Path $keepFolder $relative
        if (-not (Test-Path -LiteralPath $keep)) { throw "Missing retained file: $relative" }
        $hash = (Get-FileHash -LiteralPath $duplicate.FullName -Algorithm SHA256).Hash
        if ($hash -ne (Get-FileHash -LiteralPath $keep -Algorithm SHA256).Hash) { throw "Evaluation files differ: $relative" }
        $recorded = $comparison.candidate_files.PSObject.Properties[$relative.Replace('\','/')].Value
        if ($hash.ToLowerInvariant() -ne $recorded.sha256) { throw 'Evaluation file changed since audit' }
    }
    # Delete only these verified files, then their empty folder; no recursive deletion.
    foreach ($duplicate in $duplicateFiles) {
        $relative = $duplicate.FullName.Substring($workspaceRoot.Length + 1)
        $keepRelative = 'saved_models\test_evaluation_round_035\' + $duplicate.Name
        Remove-IdenticalFile $relative $keepRelative
    }
    Remove-Item -LiteralPath $duplicateFolder
    Remove-IdenticalFile 'reports/dataset_class_distribution1.csv' 'reports/dataset_class_distribution.csv'
    Remove-IdenticalFile 'saved_models/dataset_class_distribution.csv' 'reports/dataset_class_distribution.csv'

    Archive-WorkspaceItem 'venv' 'archive/legacy_environment/venv'
    Archive-WorkspaceItem 'dataset_backup_20260922_173524' 'archive/legacy_datasets/dataset_backup_20260922_173524'
    Archive-WorkspaceItem 'dataset_backup_20260929_chili_archive' 'archive/legacy_datasets/dataset_backup_20260929_chili_archive'
    Archive-WorkspaceItem 'AgriVision_dataset.zip' 'archive/legacy_bundles/AgriVision_dataset.zip'
    Archive-WorkspaceItem 'structure.txt' 'archive/legacy_documents/structure.txt'
    Archive-WorkspaceItem 'FIFTEEN_DAY_THESIS_PLAN.md' 'archive/documents/FIFTEEN_DAY_THESIS_PLAN.md'
    Archive-WorkspaceItem 'reports/Agri_Vision_Non_YOLO_Architecture_Review.docx' 'archive/documents/Agri_Vision_Non_YOLO_Architecture_Review.docx'
    Archive-WorkspaceItem 'reports/Agri_Vision_Non_YOLO_Architecture_Review.pdf' 'archive/documents/Agri_Vision_Non_YOLO_Architecture_Review.pdf'
    Archive-WorkspaceItem 'reports/.document_preview/architecture_review_pages.png' 'archive/legacy_documents/architecture_review_pages.png'
    Archive-WorkspaceItem 'reports/analyze_rounds_026_035.py' 'scripts/analyze_rounds_026_035.py'
    Archive-WorkspaceItem 'reports/evaluate_round_035_non_chili.py' 'scripts/evaluate_round_035_non_chili.py'
    Archive-WorkspaceItem 'reports/create_architecture_review.py' 'scripts/create_architecture_review.py'
    Write-Output ('Completed ' + $cleanupLog.Count + ' organization actions.')
} finally {
    $cleanupLog | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Resolve-WorkspaceTarget 'reports/workspace_cleanup_20261009.json') -Encoding utf8
}
