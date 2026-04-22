# Dataset

Dataset details for the MaCVi embedded semantic segmentation challenge.

## LaRS Dataset

- Location: `/home/farms/data/lars_dataset/`
  (also at `/home/farms/data/datasets/lars_dataset/` — same data)
- Splits: train 2605 images, val 198 images, test 1205 images
- Test set has NO segmentation masks (challenge server evaluates test)
- `img_dir`: `{split}/images/`
- `ann_dir`: `{split}/semantic_masks/`

## Symlink Tree

A symlink tree at `/home/farms/lars_linked/` is required by the training
scripts. It must be rebuilt each container session (not persistent across
container restarts). The `run_phase2.sh` and `run_phase2_5.sh` scripts
rebuild it automatically if missing.

## External Data

External data is allowed by the challenge rules but must be disclosed in the
final report. Phase 3 used several external datasets as augmentation sources;
see `../common/external-dataset-analysis/` for the analysis.

Available external datasets (read-only mounts under `/home/farms/data/datasets/`):
- `MULTIAQUA_night/`
- `ROSEBUD/`
- `Tampere-WaterSeg/`
- `orca_tech_water_segmentation/`
- `water_scenes/`
