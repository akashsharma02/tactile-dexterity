# Main training script - trains distributedly accordi
import os
import hydra

import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies.ddp import DDPStrategy

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm 

# Custom imports 
from tactile_dexterity.datasets.utils.dataloaders import get_dataloaders
from tactile_dexterity.learners.initialize_learner import init_learner
from tactile_dexterity.utils.logger import Logger

def train(fabric, cfg) -> None:
    # It looks at the datatype type and returns the train and test loader accordingly
    train_loader, test_loader, _ = get_dataloaders(cfg)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Initialize the learner - looks at the type of the agent to be initialized first
    learner = init_learner(cfg, fabric)

    best_loss = torch.inf 

    # Logging
    pbar = tqdm(total=cfg.train_epochs)
    # Initialize logger (wandb)
    if cfg.logger and fabric.global_rank == 0:
        hydra_run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        wandb_exp_name = '-'.join(hydra_run_dir.split('/')[-2:])
        logger = Logger(cfg, wandb_exp_name, out_dir=hydra_run_dir)

    # Start the training
    for epoch in range(cfg.train_epochs):
        # Train the models for one epoch
        train_loss = learner.train_epoch(train_loader)


        pbar.set_description(f'Epoch {epoch}, Train loss: {train_loss:.5f}, Best loss: {best_loss:.5f}')
        pbar.update(1) # Update for each batch

        # Logging
        if logger and epoch % cfg.log_frequency == 0:
            logger.log({'epoch': epoch, 'train loss': train_loss})

        # Testing and saving the model
        if epoch % cfg.save_frequency == 0: 
            learner.save(cfg.checkpoint_dir, model_type='latest') # Always save the latest encoder
            # Test for one epoch
            if not cfg.self_supervised:
                test_loss = learner.test_epoch(test_loader)
            else:
                test_loss = train_loss # In BYOL (for ex) test loss is not important

            # Get the best loss
            if test_loss < best_loss:
                best_loss = test_loss
                learner.save(cfg.checkpoint_dir, model_type='best')

            # Logging
            pbar.set_description(f'Epoch {epoch}, Test loss: {test_loss:.5f}')
            if logger:
                logger.log({'epoch': epoch,
                                'test loss': test_loss})
                logger.log({'epoch': epoch,
                                'best loss': best_loss})

    pbar.close()

@hydra.main(version_base=None,config_path='tactile_dexterity/configs', config_name = 'train')
def main(cfg : DictConfig) -> None:

    fabric = Fabric() 
    fabric.seed_everything(42)
    fabric.launch()
    train(fabric, cfg)
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()