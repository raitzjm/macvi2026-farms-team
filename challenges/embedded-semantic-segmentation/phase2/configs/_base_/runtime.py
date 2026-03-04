"""
Base runtime config for Phase 2.
mmseg 1.x (mmengine runner) API.

WandB vis backend is configured here with a placeholder run name.
Override `vis_backends` and `visualizer` in each model config with the
correct run name ({phase}_{model}).

WANDB_API_KEY must be set before training:
    source challenges/embedded-semantic-segmentation/phase2/scripts/setup_wandb.sh
WANDB_DIR must be set to an outputs dir NOT on PYTHONPATH (handled in run_phase2.sh).
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
            name='phase2_placeholder',   # overridden per model
        )),
]

visualizer = dict(
    type='SegLocalVisualizer',
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
    # draw=True: log prediction overlays to WandB at every val (interval=250)
    visualization=dict(type='SegVisualizationHook', draw=True, interval=250),
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
