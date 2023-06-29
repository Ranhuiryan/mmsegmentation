import argparse
import datetime
import os.path as osp
import subprocess
from mmcv.utils import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Staged train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--frozen_stages_schedule',
        type=int,
        nargs='+',
        default=[-1],
        help='define frozen_stages at each stage')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='enable verbose mode (print config at each stage only)')
    args = parser.parse_args()
    return args

def get_lr(cfg, current_iter: int, base_lr: float):
    progress = current_iter
    max_progress = cfg.runner.max_iters
    coeff = (1 - progress / max_progress)**cfg.lr_config.power
    return (base_lr - cfg.lr_config.min_lr) * coeff + cfg.lr_config.min_lr

def staged_train():
    args = parse_args()
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = Config.fromfile(args.config)
    base_lr = cfg.optimizer.lr
    max_iters = cfg.runner.max_iters
    work_dir = osp.join('./work_dirs', time_stamp + "_" + osp.splitext(osp.basename(args.config))[0])

    assert cfg.lr_config.policy == "poly", "Only support poly policy"

    frozen_stages_schedule = args.frozen_stages_schedule
    assert max(frozen_stages_schedule) <= cfg.model.backbone.num_stages

    for i, frozen_stages in enumerate(frozen_stages_schedule):
        # if i < len(frozen_stages_schedule) - 2:
        #     max_iters_at_this_stage = int(max_iters * (1 - 0.5 ** (i+1)))
        # elif i == len(frozen_stages_schedule) - 2:
        #     if max_iters_at_this_stage > max_iters * 0.9:
        #         max_iters_at_this_stage = int(max_iters * (1 - 0.5 ** (i+1)))
        #     else:
        #         max_iters_at_this_stage = int(max_iters * 0.9)
        # else:
        #     max_iters_at_this_stage = max_iters
        
        if i < len(frozen_stages_schedule) - 1:
            max_iters_at_this_stage = int(max_iters * (1 - 0.5 ** (i+1)))
            # max_iters_at_this_stage = int(max_iters * (i+1) / len(frozen_stages_schedule))
        else:
            max_iters_at_this_stage = max_iters

        min_lr = get_lr(cfg, max_iters_at_this_stage, base_lr)
        print(f"Stage {i}: {max_iters_at_this_stage} iters, frozen_stages={frozen_stages}, min_lr={min_lr}")

        if args.verbose:
            subprocess.run(["python", "tools/print_config.py", args.config, "--cfg-options", f"model.backbone.frozen_stages={frozen_stages}", f"runner.max_iters={max_iters_at_this_stage}", f"lr_config.min_lr={min_lr}"])
        else:    
            if i > 0:
                subprocess.run(["python", "tools/train.py", args.config, "--work-dir", work_dir, "--cfg-options", f"model.backbone.frozen_stages={frozen_stages}", f"runner.max_iters={max_iters_at_this_stage}", f"lr_config.min_lr={min_lr}", "--gpu-id", f"{args.gpu_id}", "--resume-from", f"{work_dir}/latest.pth"])
            else:
                subprocess.run(["python", "tools/train.py", args.config, "--work-dir", work_dir, "--cfg-options", f"model.backbone.frozen_stages={frozen_stages}", f"runner.max_iters={max_iters_at_this_stage}", f"lr_config.min_lr={min_lr}", "--gpu-id", f"{args.gpu_id}"])

if __name__ == '__main__':
    staged_train()