import torch
import torch.nn as nn
import os
from data import get_stl10_dataloader, sample_mask
from models.ijepa import IJEPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hyperparam = {
    # Data
    "image_size": 96,
    "patch_size": 8,
    "num_patches": 144,        
    "batch_size": 32,

    # ViT-Small
    "embed_dim": 384,
    "num_heads": 6,
    "mlp_dim": 1536,
    "num_layers": 12,

    # Predictor
    "pred_num_layers": 6,
    "pred_embed_dim": 384,

    # EMA
    "momentum": 0.996,

    # Masking
    "target_blocks": 4,
    "target_scale": (0.15, 0.2),
    "context_scale": 0.85,

    # Optimiser
    "peak_lr": 3e-5,           # 1e-3 * (64/2048)
    "base_lr": 3e-6,           # peak_lr / 10 
    "min_lr": 3e-8,            # 1e-6 * (64/2048)
    "weight_decay_start": 0.04,
    "weight_decay_end": 0.4,
    "warmup_epochs": 15,
    "num_epochs": 100,

    # Misc
    "save_every": 10,
    "checkpoint_dir": "checkpoints",
}


def main():
    os.makedirs(hyperparam["checkpoint_dir"], exist_ok=True)
    dataloader = get_stl10_dataloader(hyperparam["batch_size"], hyperparam['image_size'])
    model = IJEPA(hyperparam['image_size'], hyperparam['patch_size'], hyperparam['embed_dim'], hyperparam['num_heads'], hyperparam['mlp_dim'], hyperparam['num_layers'], hyperparam['momentum'], hyperparam['pred_num_layers']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparam["base_lr"], weight_decay=hyperparam["weight_decay_start"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparam["num_epochs"] - hyperparam["warmup_epochs"], eta_min=hyperparam["min_lr"])
    scaler = torch.amp.GradScaler('cuda')
    start_epoch = 0
    if os.path.exists(hyperparam["checkpoint_dir"]):
        checkpoints = [f for f in os.listdir(hyperparam["checkpoint_dir"]) if f.endswith('.pt')]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
            checkpoint = torch.load(os.path.join(hyperparam["checkpoint_dir"], latest))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    for epoch in range(start_epoch, hyperparam['num_epochs']):
        if epoch < hyperparam['warmup_epochs']:
            lr = hyperparam['base_lr'] + (hyperparam['peak_lr'] - hyperparam['base_lr']) * (epoch / hyperparam['warmup_epochs'])
            for group in optimizer.param_groups:
                group['lr'] = lr
        else:
            scheduler.step()
        for group in optimizer.param_groups:
            group['weight_decay'] = hyperparam['weight_decay_start'] + (hyperparam['weight_decay_end'] - hyperparam['weight_decay_start']) * (epoch / hyperparam['num_epochs'])
        
        epoch_loss = 0.0
        num_batches = 0
        for batch, _ in dataloader:
            train = batch.to(device)
            cindex, tindex = sample_mask(int(hyperparam['num_patches']**0.5), hyperparam['target_blocks'], hyperparam['target_scale'], hyperparam['context_scale'])
            cindex = cindex.to(device)
            tindex = [t.to(device) for t in tindex]
            loss = torch.tensor(0.0, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, target = model(train, cindex, tindex)
                for predb, block_index in zip(pred, tindex):
                    t = nn.functional.layer_norm(target[:, block_index, :].float(), target[:, block_index, :].shape[-1:])
                    loss += ((predb.float() - t)**2).sum()
                loss /= hyperparam['target_blocks']
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_target()
            epoch_loss += loss.item()
            num_batches += 1
        if epoch % hyperparam['save_every'] == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss,}, f"{hyperparam['checkpoint_dir']}/checkpoint_epoch_{epoch}.pt")
        print(f"Epoch {epoch}/{hyperparam['num_epochs']} | Loss: {epoch_loss/num_batches:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")





if __name__ == "__main__":
    main()