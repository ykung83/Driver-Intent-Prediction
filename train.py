import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml
import wandb
import argparse

# import sys
# sys.path.append("./models")

from models.utils import *
from models.transformer import CustomTransformer
from dataloader.dataset_b4c import B4CDataset

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', default="config/transformer_all.yaml",
                    help="Path to the model config file")
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

seed=42
def main(args):
    # Set up Weights & Biases
    # wandb.init(project="transformer_training_example")
    cfg=None
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # Hyperparameters
    BATCH_SIZE  = cfg['OPTIMIZATION']['BATCH_SIZE_PER_GPU']
    LR          = cfg['OPTIMIZATION']['LR']
    NUM_EPOCHS  = cfg['OPTIMIZATION']['NUM_EPOCHS']
    BETA1, BETA2 = cfg['OPTIMIZATION']['BETA1'], cfg['OPTIMIZATION']['BETA2']

    # Initialize the DataLoader
    dataset = B4CDataset(cfg, split="train", create_dataset=False) # Set to True to generate new dataset
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize your Transformer model
    model = CustomTransformer()
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    setup_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(BETA1, BETA2))

    import pdb; pdb.set_trace()
    # Training loop
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (data, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Log training progress to wandb
            wandb.log({"epoch": epoch, "batch_idx": batch_idx, "loss": loss.item()})

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)