# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp

from mmcv.utils import TORCH_VERSION
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
from sacred import Experiment

from mmdet3d.experiment import experiment


@experiment.ex.capture
def log_run(tag, val, _run):
    # _run.result = 100
    _run.log_scalar(tag, val)


@HOOKS.register_module()
class SacredLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(SacredLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'sacred_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
                log_run(tag, val)

    @master_only
    def after_run(self, runner):
        self.writer.close()