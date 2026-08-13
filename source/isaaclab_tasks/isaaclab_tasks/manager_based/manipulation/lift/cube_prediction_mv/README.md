# cube_prediction_mv/ — not tracked in git

This directory holds one trained checkpoint (`mv_vl_0.0079.pth`, ~97MB) from the
same cube-pose-prediction experiment family as the outer repo's `computer_vision/`
directory. Not committed here for the same reason: the project's GitHub org is on
the free tier (Git LFS capped at 1GB storage/month), and this file alone is right
at GitHub's 100MB hard per-file block.

**Full backup location (2026-08-14):** Virga HPC,
`/datasets/work/hri-fyp2025s1-2903/work/model_checkpoints_backup/cube_prediction_mv_isaaclab_wip/cube_prediction_mv/`

Verified byte-identical to the copy on this workstation at backup time
(100,784,466 bytes, matched exactly on both ends).

To retrieve a fresh copy elsewhere:

```bash
scp -r <virga-user>@virga.hpc.csiro.au:/datasets/work/hri-fyp2025s1-2903/work/model_checkpoints_backup/cube_prediction_mv_isaaclab_wip/cube_prediction_mv <destination>
```

See also `computer_vision/README.md` in the outer `hri-pl-frm-mvvd` repo, which
documents the larger related checkpoint collection backed up the same way.
