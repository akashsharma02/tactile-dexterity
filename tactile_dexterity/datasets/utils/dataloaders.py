import hydra
import torch
import torch.utils.data as data 

from omegaconf import DictConfig

# Script to return dataloaders
def get_dataloaders(cfg : DictConfig):
    # Load dataset - splitting will be done with random splitter
    
    if 'image' in cfg.learner_type:
        dataset = hydra.utils.instantiate(cfg.dataset,
                                          data_path = cfg.data_dir)
    else:
        dataset = hydra.utils.instantiate(cfg.dataset,
                                        data_path = cfg.data_dir,
                                        tactile_img_size = cfg.tactile_image_size) # This should be named this way 
        

    train_dset_size = int(len(dataset) * cfg.train_dset_split)
    test_dset_size = len(dataset) - train_dset_size

    # Random split the train and validation datasets
    train_dset, test_dset = data.random_split(dataset, 
                                             [train_dset_size, test_dset_size],
                                             generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = data.DataLoader(train_dset, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers)
    test_loader = data.DataLoader(test_dset, batch_size=cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers)

    return train_loader, test_loader, dataset
    