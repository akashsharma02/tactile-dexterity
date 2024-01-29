import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

from .byol import BYOLLearner
from .vicreg import VICRegLearner
from .behavior_cloning import ImageTactileBC

from tactile_dexterity.utils.augmentations import get_tactile_augmentations, get_vision_augmentations
from tactile_dexterity.utils.constants import VISION_IMAGE_MEANS, VISION_IMAGE_STDS, TACTILE_IMAGE_MEANS, TACTILE_IMAGE_STDS
from tactile_dexterity.models import BYOL, VICReg, create_fc 

def init_learner(cfg, fabric):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, fabric)
    elif 'tactile' in cfg.learner_type:
        return init_tactile_byol(
            cfg,
            fabric,
            aug_stat_multiplier=cfg.learner.aug_stat_multiplier,
            byol_in_channels=cfg.learner.byol_in_channels,
            byol_hidden_layer=cfg.learner.byol_hidden_layer
        )
    elif cfg.learner_type == 'image_byol':
        return init_image_byol(cfg, fabric)
    elif cfg.learner_type == 'vicreg':
        return init_tactile_vicreg(
            cfg,
            fabric,
            sim_coef=cfg.learner.sim_coef,
            std_coef=cfg.learner.std_coef,
            cov_coef=cfg.learner.cov_coef)
    
    return None

def init_tactile_vicreg(cfg, fabric, sim_coef, std_coef, cov_coef):

    backbone = hydra.utils.instantiate(cfg.encoder)
    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS,
        img_stds = TACTILE_IMAGE_STDS,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )

    # Initialize the vicreg projector
    projector = create_fc(
        input_dim = cfg.encoder.out_dim,
        output_dim = 8192,
        hidden_dims = [8192],
        use_batchnorm = True
    )

    # Initialize the vicreg wrapper 
    vicreg_wrapper = VICReg(
        backbone = backbone,
        projector = projector, 
        augment_fn = augment_fn, 
        sim_coef = sim_coef, 
        std_coef = std_coef, 
        cov_coef = cov_coef
    )

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg. optimizer,
                                        params = vicreg_wrapper.parameters())
    vicreg_wrapper, optimizer = fabric.setup(vicreg_wrapper, optimizer)

    learner = VICRegLearner(
        vicreg_wrapper = vicreg_wrapper, 
        optimizer = optimizer,
        fabric=fabric,
    )                 

    return learner

def init_tactile_byol(cfg, fabric, aug_stat_multiplier=1, byol_in_channels=3, byol_hidden_layer=-2):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder)

    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS*aug_stat_multiplier,
        img_stds = TACTILE_IMAGE_STDS*aug_stat_multiplier,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.tactile_image_size,
        augment_fn = augment_fn,
        hidden_layer = byol_hidden_layer,
        in_channels = byol_in_channels
    )
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    byol, optimizer = fabric.setup(byol, optimizer)
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        fabric = fabric,
        byol_type = 'tactile'
    )

    return learner

def init_image_byol(cfg, fabric):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.vision_image_size,
        augment_fn = augment_fn
    )
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    byol, optimizer = fabric.setup(byol, optimizer) 
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        fabric=fabric,
        byol_type = 'image'
    )

    return learner


def init_bc(cfg, fabric):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder)
    tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder)
    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer)

    optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    image_encoder, tactile_encoder, last_layer = fabric.setup(image_encoder, tactile_encoder, last_layer, optimizer)

    learner = ImageTactileBC(
        image_encoder = image_encoder, 
        tactile_encoder = tactile_encoder,
        last_layer = last_layer,
        optimizer = optimizer,
        fabric=fabric,
        loss_fn = 'mse',
        representation_type='all'
    )
    
    return learner