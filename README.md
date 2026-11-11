# Wheel-Based Dual-Arm Braking Research

This repository contains the planning records and simulation code for the
controlled-braking and payload-stability study of a wheeled dual-arm robot.

## Layout

- `docs/`: project plan, stage plans, and decision records.
- `simulation/rby1/`: MuJoCo model builders, stopping references, checks, and
  analysis scripts.
- `outputs/`: curated experiment reports and audit inputs. The latest impact
  pair includes its measurement traces and the offline comparison dashboard.
  The remaining bulk trajectories, source PDFs, and local runs stay in the
  local archive.

The [project plan](docs/PROJECT_PLAN.md) tracks current status. The
[S3 handoff and evidence index](docs/plans/34-s3-mature-stop-baselines.md#阶段交接与证据索引)
links the tray and moving-load results, explains their scope, and gives the
command to recompute the saved impact comparison. Download the
[offline dashboard](outputs/rby1-research/slider-dashboard/slider-dashboard.html)
and open it locally to replay the trials.
