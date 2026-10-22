# EfficientDet parallel screening

4gmail runs trials 12-16; 71gmail runs trials 1-11. Each trial keeps the original five-round screening budget. The two accounts save independently in their own Agri Vision folders. No run is launched by this preparation.

On 4gmail open AgriVision_GridSearch_EfficientDet.ipynb, select T4, then Run all. It now invokes parallel_effdet.screen with first=12 and last=16. Reopen the updated Drive notebook before running; an older open tab may still contain the original full-search command.

On 71gmail use AgriVision_EfficientDet_Trials01_11_71gmail.ipynb on the next restart. The already running original process is unaffected by uploaded notebook changes. It must not continue to trial 12 while 4gmail owns trials 12-16. Do not interrupt an active round just to switch notebooks; switch after the session stops or a round is committed.

The new driver uses unchanged tuning.search configurations, preflight, ensure_trial, checkpoints and metrics. It routes training to the existing continuation.runner. The original code ZIP and optimized extension ZIP retain their SHA256 values. Both notebooks check the common initial.pt hash before running. The copied execution profile and original extension fingerprint must match the benchmarked T4 software environment.

The bounded job writes parallel_12_16_assignment.json, parallel_12_16_status.json and parallel_12_16_results.csv on 4gmail (01_11 equivalents on 71gmail). Existing full-search plan.json, search_status.json and screening_results.csv are not replaced by this job. Read the parallel status plus the active trial's status.json and latest.json to monitor it.

Four GPU hours are a partial-session budget, not a promise to complete all 25 rounds of the assigned five trials. Interrupted rounds repeat from the last committed global checkpoint; completed trials are skipped after identity validation.

After both ranges finish, copy trials 12-16, their configs and per-round metrics to the canonical 71gmail model folder. Verify common initialization, execution profile, config/environment identities, checkpoint hashes and five committed round records per trial. Preserve originals and reject conflicting trial copies. The absolute initial-checkpoint paths are identical because both accounts use /content/drive/MyDrive/Agri Vision/grid_search/grid_v1_20261017/efficientdet_d0/initial.pt. Then resume the original 71gmail continuation notebook: its existing controller ranks all 16 trials, confirms the global top two and performs the fresh 30-round final run. Do not select finalists independently on each subset.

Validation: three CPU tests passed for exact range assignment, five-round-only budgets, no local finalist selection and rejection of an overlapping range. Notebook code cells are parsed when packaged and outputs are cleared. No GPU training was started locally.
