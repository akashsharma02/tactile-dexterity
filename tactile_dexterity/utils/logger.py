import os
import wandb

from omegaconf import DictConfig, OmegaConf
from lightning.fabric.utilities.rank_zero import rank_zero_only

# Class for the wandb logger
class Logger:
    def __init__(self, cfg : DictConfig, exp_name:str, out_dir:str) -> None:
        # Initialize the wandb experiment
        self.wandb_logger = wandb.init(project="tactile_dexterity", 
                                       name=exp_name,
                                       config = OmegaConf.to_container(cfg, resolve=True), settings=wandb.Settings(start_method="thread"))
        self.logger_file = os.path.join(out_dir, 'train.log')

    @rank_zero_only
    def log(self, msg):
        if type(msg) is dict:
            self.wandb_logger.log(msg)
            # if 'best loss' in msg:
            #     self.tb_logger.add_scalar('Best Loss', msg['best loss'], msg['epoch'])
            # if 'train loss' in msg:
            #     self.tb_logger.add_scalar('Train Loss', msg['train loss'], msg['epoch'])
            # if 'test loss' in msg:
            #     self.tb_logger.add_scalar('Test Loss', msg['test loss'], msg['epoch'])

        with open(self.logger_file, 'a') as f:
            f.write('{}\n'.format(msg))