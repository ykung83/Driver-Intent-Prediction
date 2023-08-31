import os
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import yaml
import wandb
import argparse

from tqdm import tqdm

# import sys
# sys.path.append("./models")

from models.utils import *
from models.transformer import TransformerModel
from dataloader.dataset_b4c import B4CDataset

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', default="config/transformer_all.yaml",
                    help="Path to the model config file")
parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
parser.add_argument('--extra_tag', type=str, default="default", help='Run for tag')
parser.add_argument('--save_dir', type=str, default="./outputs", help='directory to save checkpoints')
parser.add_argument('--use_wandb', type=str, default=False, help='log to wandb?')


seed=42

def setup_directories(save_dir, extra_tag):
    run_save_dir = join(save_dir, extra_tag)
    ckpt_save_dir = join(run_save_dir, 'ckpts')
    if not os.path.exists(run_save_dir):
        print(f'Creating run save directory {run_save_dir}')
        os.makedirs(run_save_dir)
    if not os.path.exists(ckpt_save_dir):
        print(f'Creating run save directory {ckpt_save_dir}')
        os.makedirs(ckpt_save_dir)

    return run_save_dir, ckpt_save_dir

def main(args):
    cfg=None
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    save_dir = args.save_dir
    extra_tag = args.extra_tag
    run_save_dir, ckpt_save_dir = setup_directories(save_dir, extra_tag)
    
    use_wandb = args.use_wandb

    if use_wandb:
        model_architecture = cfg['model']
        # Set up Weights & Biases
        wandb.init(project="driver-intent-learning",
            name=f'{model_architecture}_{extra_tag}',
            config={
                "model": model_architecture,
                "lr": cfg['OPTIMIZATION']['LR'],
                "epochs": cfg['OPTIMIZATION']['NUM_EPOCHS'],
                "batch_size": cfg['OPTIMIZATION']['BATCH_SIZE_PER_GPU'],
                "usekfold": cfg['DATALOADER_CONFIG']['USE_KFOLD'],
            }
        )

    # Hyperparameters
    BATCH_SIZE  = cfg['OPTIMIZATION']['BATCH_SIZE_PER_GPU']
    LR          = cfg['OPTIMIZATION']['LR']
    NUM_EPOCHS  = cfg['OPTIMIZATION']['NUM_EPOCHS']
    BETA1, BETA2 = cfg['OPTIMIZATION']['BETA1'], cfg['OPTIMIZATION']['BETA2']
    USE_KFOLD       = cfg['DATALOADER_CONFIG']['USE_KFOLD']
    DECAY_RATE      = cfg['OPTIMIZATION']['DECAY_RATE']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize your Transformer model
    model = TransformerModel(cfg['MODEL_CONFIG'], device=device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    setup_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(BETA1, BETA2))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)

    # Initialize the DataLoader
    if USE_KFOLD:
        dataset = B4CDataset(cfg, split="5_fold", create_dataset=False) # Set to True to generate new dataset
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=True)

        # Generate 5-fold cross validation datasets deterministically
        generator1 = torch.Generator().manual_seed(42)
        split_len = len(dataset) #5
        fold_datasets_list = random_split(dataset, [split_len]*5, generator=generator1)
        fold_dataloaders_list = [DataLoader(fold_dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=True) for fold_dataset in fold_datasets_list]
    else:
        train_dataset = B4CDataset(cfg, split="train", create_dataset=False) # Set to True to generate new dataset
        train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=False)

        val_dataset = B4CDataset(cfg, split="val", create_dataset=False) # Set to True to generate new dataset
        val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, batch_size=BATCH_SIZE, shuffle=False)

    for name, param in model.named_parameters():
        print(name)
    import pdb; pdb.set_trace()
    # Training loop
    running_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for input_data, action_labels in tqdm(train_dataloader):
            if running_step > 100:
                break
            optimizer.zero_grad()
            input_data = torch.from_numpy(np.array(input_data)).float().to(device)
            batch_gt = torch.from_numpy(np.array(action_labels)).to(device, non_blocking=True)
            batch_preds = model(input_data)

            # Evaluate on softmax
            loss = criterion(batch_preds, batch_gt.long())

            loss.backward()
            optimizer.step()

            #Train Accuracy
            with torch.no_grad():
                max_batch_preds = torch.argmax(batch_preds, dim=-1)
                preds_masked = max_batch_preds.cpu().numpy()
                gt_masked = batch_gt.cpu().numpy()

                num_correct = np.sum(preds_masked == gt_masked)
                num_total = preds_masked.shape[0]
                accuracy = num_correct/num_total
                running_step += len(preds_masked)
            
            if use_wandb:
                # Log training progress to wandb
                wandb.log({"epoch": epoch, 
                        "step": running_step ,
                        "batch_idx": input_data[0], 
                        "loss": loss.item(),
                        "lr": lr_scheduler.get_lr()[0],
                        "train_acc": accuracy})
        print("lr ", lr_scheduler.get_last_lr()[0])
        # Decrease LR and save model checkpoint
        lr_scheduler.step()
        # torch.save(model.state_dict(), join(ckpt_save_dir, f'epoch{epoch}.pth') )
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            counter = 0
            num_correct = 0
            num_total = 0
            for input_data, action_labels in tqdm(val_dataloader):
                if counter > 100:
                    break
                optimizer.zero_grad()
                input_data = torch.from_numpy(np.array(input_data)).float().to(device)
                batch_gt = torch.from_numpy(np.array(action_labels)).to(device, non_blocking=True)
                batch_preds = model(input_data)

                loss = criterion(batch_preds, batch_gt.long())
                running_loss += loss.item()
                counter += input_data.shape[0]

                # Validation Accuracy
                max_batch_preds = torch.argmax(batch_preds, dim=-1)
                preds_masked = max_batch_preds.cpu().numpy()
                gt_masked = batch_gt.cpu().numpy()

                num_correct += np.sum(preds_masked == gt_masked)
                num_total += preds_masked.shape[0]
                accuracy = num_correct/num_total

            # Log validation progress to wandb
            print(f'Eppoch Num: {epoch} ------ average val loss: {running_loss/counter}')
            print(f'Eppoch Num: {epoch} ------ average val accuracy: {accuracy}')
            if use_wandb:
                wandb.log({"epoch": epoch, "val_acc": accuracy, "val_loss": running_loss/counter})

    if use_wandb:
        # Finish the wandb run
        wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)