import os
import torch

from .learner import Learner 

class BYOLLearner(Learner):
    def __init__(
        self,
        byol,
        optimizer,
        fabric,
        byol_type
    ):
        self.optimizer = optimizer 
        self.byol = byol
        self.fabric = fabric
        self.byol_type = byol_type # Tactile or Image

    def to(self, device):
        self.device = device 
        self.byol.to(device)

    def train(self):
        self.byol.train()

    def eval(self):
        self.byol.eval()

    def save(self, checkpoint_dir, model_type='best'):
        self.fabric.save(os.path.join(checkpoint_dir, f'byol_encoder_{model_type}.pt'), self.byol.state_dict())

    def train_epoch(self, train_loader):
        self.train() 

        # Save the train loss
        train_loss = 0.0 

        # Training loop 
        for image in train_loader: 
            self.optimizer.zero_grad()

            # Get the loss by the byol            
            loss = self.byol(image)
            train_loss += loss.item() 

            # Backprop
            self.fabric.backward(loss)
            self.optimizer.step()
            self.byol.update_moving_average() 
            print(f"train_loss: {train_loss}")

        return train_loss / len(train_loader)