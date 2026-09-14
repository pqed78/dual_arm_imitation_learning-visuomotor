# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Training Script for Dual Arm Imitation Learning.

Supports:
    - Behavior Cloning (BC)
    - Diffusion Policy
    - Action Chunking with Transformers (ACT)

Usage:
    python scripts/train.py --algo=bc --epochs=100
    python scripts/train.py --algo=diffusion --epochs=150
    python scripts/train.py --algo=act --epochs=150
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy
import copy
import random
from torch.utils.data import Sampler, Subset

class ChunkedRandomSampler(Sampler):
    """
    Groups dataset indices into large sequential chunks, shuffles the chunks, 
    and then shuffles indices within each chunk.
    This guarantees 99%+ HDF5 chunk cache hit rates by preserving spatial locality,
    preventing catastrophic 'Chunk Thrashing' caused by purely random DataLoader sampling.
    """
    def __init__(self, data_source, chunk_size=20000):
        self.data_source = data_source
        self.chunk_size = chunk_size
        
    def __iter__(self):
        indices = list(range(len(self.data_source)))
        chunks = [indices[i:i + self.chunk_size] for i in range(0, len(indices), self.chunk_size)]
        random.shuffle(chunks)
        for chunk in chunks:
            random.shuffle(chunk)
            for idx in chunk:
                yield idx

    def __len__(self):
        return len(self.data_source)
import random

def apply_gpu_augmentation(images):
    """
    Applies Color Jitter and Random Crop purely on the GPU to completely eliminate CPU/Dataloader bottleneck.
    Expects images tensor: (B, T, C, H, W) float [0, 1]
    """
    B, T, C, H, W = images.shape
    device = images.device

    # 1. Color Jitter (Probability 80%)
    if random.random() < 0.8:
        # Generate random factors per batch element (B, 1, 1, 1, 1)
        # Apply the exact same jitter to all T frames in each sequence
        b_f = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.8, 1.2)
        c_f = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.8, 1.2)
        
        # Apply brightness (multiply)
        images = images * b_f
        
        # Apply contrast (interpolate with mean)
        # mean over C, H, W (last 3 dims)
        mean = images.mean(dim=[-3, -2, -1], keepdim=True)
        images = (images - mean) * c_f + mean
        
        images = torch.clamp(images, 0.0, 1.0)
        
    # 2. Random Crop (Translation)
    pad = 4
    # Fold B and T into a 4D tensor (B*T, C, H, W) because PyTorch replicate pad requires 3D/4D tensors
    images_4d = images.view(B * T, C, H, W)
    images_padded = torch.nn.functional.pad(images_4d, (pad, pad, pad, pad), mode='replicate')
    images_padded = images_padded.view(B, T, C, H + pad * 2, W + pad * 2)
    
    # We slice uniquely per batch element, but consistently across T
    # Using a fast loop over B to slice (very fast on GPU as it just modifies tensor views)
    cropped = torch.empty_like(images)
    for i in range(B):
        top = random.randint(0, pad * 2)
        left = random.randint(0, pad * 2)
        cropped[i] = images_padded[i, :, :, top:top+H, left:left+W]
        
    return cropped

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                
    def copy_to(self, target_model):
        for name, param in target_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def parse_args():
    parser = argparse.ArgumentParser(description="Train imitation learning policies.")
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion",
        choices=["bc", "diffusion", "act"],
        help="Algorithm to train: bc, diffusion, or act.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
        help="Path to HDF5 demonstration dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/{algo}_cfg.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of CPU workers for DataLoader.")
    return parser.parse_args()


from torch.utils.tensorboard import SummaryWriter

def load_config(algo: str, custom_cfg_path: str | None) -> dict:
    if custom_cfg_path is None:
        cfg_path = os.path.join(PROJECT_ROOT, "configs", f"{algo}_cfg.yaml")
    else:
        cfg_path = custom_cfg_path

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(algo: str, cfg: dict, obs_dim: int, act_dim: int) -> torch.nn.Module:
    if algo == "bc":
        model_type = cfg.get("model_type", "mlp")
        if model_type == "rnn":
            return RNNBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=cfg.get("rnn_hidden_dim", 256),
                num_layers=cfg.get("rnn_num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            return MLPBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=cfg.get("hidden_dims", [512, 512, 256]),
                dropout=cfg.get("dropout", 0.1),
                activation=cfg.get("activation", "relu"),
            )

    elif algo == "diffusion":
        return DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
            kernel_size=cfg.get("kernel_size", 5),
            n_groups=cfg.get("n_groups", 8),
        )

    elif algo == "act":
        return ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_encoder_layers=cfg.get("num_encoder_layers", 4),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            latent_dim=cfg.get("latent_dim", 32),
            kl_weight=cfg.get("kl_weight", 10.0),
            dropout=cfg.get("dropout", 0.1),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
            temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff", 0.01),
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.algo, args.config)

    # Overrides
    epochs = args.epochs or cfg.get("epochs", 100)
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    lr = args.lr or cfg.get("learning_rate", 1.0e-4)
    weight_decay = cfg.get("weight_decay", 1.0e-5)

    print("=" * 60)
    print(f" Training Imitation Learning Policy: {args.algo.upper()}")
    print("=" * 60)
    print(f" Dataset    : {args.dataset}")
    print(f" Batch Size : {batch_size}")
    print(f" Epochs     : {epochs}")
    print(f" Device     : {args.device}")
    print(f" LR         : {lr}")
    print("=" * 60)

    # Setup directories
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard Writer
    tb_dir = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f" TensorBoard logged to: {tb_dir}")

    # Load Dataset
    pred_h = cfg.get("pred_horizon", 16) if args.algo == "diffusion" else cfg.get("chunk_size", 24)
    obs_h = cfg.get("obs_horizon", 2)

    full_dataset = DualArmDataset(
        dataset_path=args.dataset,
        algo=args.algo,
        pred_horizon=pred_h,
        obs_horizon=obs_h,
    )

    # Save normalization stats for evaluation
    stats_path = os.path.join(save_dir, "stats.pkl")
    full_dataset.save_stats(stats_path)

    # Train / Val Split (90% train, 10% val)
    # CRITICAL: We MUST use sequential split instead of random_split.
    # random_split shuffles the dataset indices globally, destroying spatial locality
    # and causing catastrophic HDF5 chunk thrashing in DataLoader workers.
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
    
    # Enable Data Augmentation for training dataset only
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.is_train = False
    train_dataset.dataset.is_train = True

    # Optimize DataLoader for lazy HDF5 loading
    num_workers = args.num_workers
    print(f" CPU Workers: {num_workers}")
    
    # Use ChunkedRandomSampler to preserve HDF5 cache locality (20k transitions per chunk)
    sampler = ChunkedRandomSampler(train_dataset, chunk_size=20000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None
    )

    # Instantiate Model
    model = build_model(args.algo, cfg, full_dataset.obs_dim, full_dataset.act_dim)
    model.to(args.device)
    raw_model = model  # Keep reference to raw model for deepcopy, EMA, and saving

    # Optimizer & Scheduler (Use raw_model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # PyTorch AMP Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize EMA (Use raw_model)
    ema = EMAModel(raw_model)

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Model] Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                raw_model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                if "ema_shadow" in checkpoint:
                    # Clean _orig_mod. prefix from keys if they were accidentally saved with them
                    cleaned_shadow = {}
                    for k, v in checkpoint["ema_shadow"].items():
                        clean_k = k.replace("_orig_mod.", "")
                        cleaned_shadow[clean_k] = v
                    ema.shadow = cleaned_shadow
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                if "val_loss" in checkpoint:
                    best_val_loss = checkpoint["val_loss"]
                print(f"  --> Resumed from epoch {start_epoch - 1}, best_val_loss {best_val_loss:.5f}")
            else:
                raw_model.load_state_dict(checkpoint)
                print(f"  --> Resumed raw weights only.")
        else:
            print(f"[Warning] Checkpoint not found: {args.resume}. Starting from scratch.")

    # Apply Torch Compile for 10-20% speedup on GPU AFTER loading weights
    try:
        model = torch.compile(raw_model)
        print("[Train] torch.compile() applied successfully for speedup.")
    except Exception as e:
        print(f"[Train] torch.compile() failed or not supported: {e}")

    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"[Model] {args.algo.upper()} created with {total_params:,} trainable parameters.")

    # Training Loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            if "rgb_image" in batch:
                # Cast uint8 to float32 on GPU (Saves 75% PCIe bandwidth)
                batch["rgb_image"] = batch["rgb_image"].float() / 255.0
                # Apply ultra-fast GPU augmentation!
                batch["rgb_image"] = apply_gpu_augmentation(batch["rgb_image"])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model.compute_loss(batch)
                loss = loss_dict["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step EMA
            ema.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            global_step += 1
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            # Log step-wise training loss
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation (using EMA model)
        # Create a temporary model copy for EMA validation
        ema_val_model = copy.deepcopy(raw_model)
        ema.copy_to(ema_val_model)
        ema_val_model.eval()
        
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                if "rgb_image" in batch:
                    batch["rgb_image"] = batch["rgb_image"].float() / 255.0
                    
                with torch.cuda.amp.autocast():
                    loss_dict = ema_val_model.compute_loss(batch)
                val_loss_sum += loss_dict["loss"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f} | EMA Val Loss: {avg_val_loss:.5f}")

        # Log epoch-wise metrics to TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Prepare Checkpoint Dictionary
        checkpoint_dict = {
            "epoch": epoch,
            "algo": args.algo,
            "config": cfg,
            "obs_dim": full_dataset.obs_dim,
            "act_dim": full_dataset.act_dim,
            "model_state_dict": ema_val_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_shadow": ema.shadow,
            "val_loss": best_val_loss,
        }

        # Save Best Checkpoint (Save EMA weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            
            torch.save(checkpoint_dict, best_ckpt_path)
            print(f"  --> Saved new best model to {best_ckpt_path} (Val Loss: {best_val_loss:.5f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_interval", 20) == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint_dict, ckpt_path)

    writer.close()
    print("\n[Training Complete]")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
