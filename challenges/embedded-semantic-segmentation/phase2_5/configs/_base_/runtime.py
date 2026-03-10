"""
Base runtime config for Phase 2.5.
mmseg 1.x (mmengine runner) API.

Fix 4: Uses MacViSegVisualizer instead of SegLocalVisualizer to overlay
       ignore areas (GT==255) on the predicted segmentation image in WandB.

WandB vis backend is configured here with a placeholder run name.
Override `vis_backends` and `visualizer` in each model config with the
correct run name ({phase}_{model}).

WANDB_API_KEY must be set before training:
    source challenges/embedded-semantic-segmentation/phase2_5/scripts/setup_wandb.sh
WANDB_DIR must be set to an outputs dir NOT on PYTHONPATH (handled in run_phase2_5.sh).
"""

default_scope = 'mmseg'

# ── Visualization + WandB ──────────────────────────────────────────────────────
# Each model config overrides vis_backends with the correct run name.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            entity='little_farms',
            project='embedded_segmentation_macvi2026',
            name='phase2_5_placeholder',   # overridden per model
        )),
]

# Fix 4: MacViSegVisualizer overlays ignore areas on predicted images in WandB.
visualizer = dict(
    type='MacViSegVisualizer',
    vis_backends=vis_backends,
    alpha=0.8)

# ── Default hooks ──────────────────────────────────────────────────────────────
# checkpoint.interval is overridden per model config.
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,          # overridden per model
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # draw=True: log prediction overlays to WandB. interval=5 → every 5th val
    # image logged (~40 of 198 per eval). interval=250 would NEVER fire because
    # (batch_idx+1) % 250 == 0 requires batch_idx=249, but val has only 198 imgs.
    visualization=dict(type='SegVisualizationHook', draw=True, interval=5),
)

# ── Runtime ───────────────────────────────────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

randomness = dict(seed=42)
log_level = 'INFO'
load_from = None
resume = False
