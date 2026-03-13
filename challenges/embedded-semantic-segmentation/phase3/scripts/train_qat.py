"""
Custom QAT training script to bypass MMRazor tracing issues.
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmrazor.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Train a razor model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key=\"[a,b]\" or key=a,b '
        'It also allows nested list/tuple values, e.g. key=\"[(a,b),(c,d)]\" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    register_all_modules(False)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # ── Monkey patch Runner to skip config logging if it fails ──
    import mmengine.runner.runner as runner_mod
    original_log_env = runner_mod.Runner._log_env
    def patched_log_env(self, env_cfg):
        try:
            original_log_env(self, env_cfg)
        except Exception as e:
            self.logger.warning(f"Failed to log env/config: {e}")
    runner_mod.Runner._log_env = patched_log_env

    # ── Build Model ──
    from mmseg.apis import init_model
    # We need to extract the actual architecture from the MMRazor config or just use the mmseg model
    # For simplicity, let's just init the model from the checkpoint
    
    # Extract info from cfg.model
    if 'architecture' in cfg.model:
        # It's an MMRazor config
        m_cfg = cfg.model.architecture
        if '_scope_' in m_cfg:
            del m_cfg['_scope_']
        # We need a dummy config file for init_model if we don't want to recreate it
        # Actually, let's just build it via MODELS registry
        from mmseg.registry import MODELS
        model = MODELS.build(m_cfg)
        
        # Load high-precision checkpoint
        from mmengine.runner import load_checkpoint
        load_checkpoint(model, cfg.model.float_checkpoint, map_location='cpu')
    else:
        # Standard mmseg model
        from mmseg.registry import MODELS
        model = MODELS.build(cfg.model)

    # ── Manual QAT Preparation ──
    from prepare_qat_model import wrap_model_qat
    model = wrap_model_qat(model)
    print("Manual QAT preparation complete.")

    # Update cfg.model to use our prepared model
    # We can't easily put the model object back into cfg if Runner.from_cfg builds it
    # So we will pass the model to the Runner constructor directly.

    # ── Build Runner ──
    # We need to handle datasets/dataloaders correctly
    runner = Runner(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=cfg.train_dataloader,
        val_dataloader=cfg.val_dataloader,
        train_cfg=dict(type='IterBasedTrainLoop', max_iters=cfg.train_cfg.max_iters, val_interval=cfg.train_cfg.val_interval),
        val_cfg=cfg.val_cfg,
        val_evaluator=cfg.val_evaluator,
        optim_wrapper=cfg.optim_wrapper,
        param_scheduler=cfg.param_scheduler,
        default_hooks=cfg.default_hooks,
        visualizer=cfg.visualizer,
        default_scope=cfg.default_scope,
        env_cfg=cfg.env_cfg,
        log_level=cfg.log_level,
    )
    runner.train()

if __name__ == '__main__':
    main()
