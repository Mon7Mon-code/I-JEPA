import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.ijepa import IJEPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparam = {
    "image_size": 96,
    "patch_size": 8,
    "embed_dim": 384,
    "num_heads": 6,
    "mlp_dim": 1536,
    "num_layers": 12,
    "pred_num_layers": 6,
    "momentum": 0.996,

    # Linear probe
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": 50,
    "lr": 0.4,
    "weight_decay": 0.0,
    "step_size": 15,
    "gamma": 0.1,

    # Checkpoint
    "checkpoint_path": "checkpoints/checkpoint_epoch_90.pt",
}

def get_train_loader(batch_size, image_size=96):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    dataset = datasets.STL10(root='data', split='train', download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def get_test_loader(batch_size, image_size=96):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    dataset = datasets.STL10(root='data', split='test', download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def main():
    train_set = get_train_loader(hyperparam['batch_size'], hyperparam['image_size'])
    checkpoint = torch.load(hyperparam['checkpoint_path'], map_location=device)
    model = IJEPA(hyperparam['image_size'], hyperparam['patch_size'], hyperparam['embed_dim'], hyperparam['num_heads'], hyperparam['mlp_dim'], hyperparam['num_layers'], hyperparam['momentum'], hyperparam['pred_num_layers']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    context_enc = model.context
    context_enc.eval()
    for param in context_enc.parameters():
        param.requires_grad = False

    linear_probe = nn.Linear(hyperparam['embed_dim'], hyperparam['num_classes']).to(device)
    optimizer = torch.optim.SGD(linear_probe.parameters(), lr=hyperparam['lr'], momentum=0.9, weight_decay=hyperparam['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparam['step_size'], gamma=hyperparam['gamma'])
    loss_func = nn.CrossEntropyLoss()


    for epoch in range(hyperparam['num_epochs']):
        linear_probe.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch, labels  in train_set:
            batch = batch.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = context_enc(batch).mean(dim=1)
            pred = linear_probe(features)
            loss = loss_func(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1 
        scheduler.step()
        print(f"Epoch {epoch+1}/{hyperparam['num_epochs']} | Loss: {epoch_loss/num_batches:.4f}")
    
    
    test_set = get_test_loader(hyperparam['batch_size'], hyperparam['image_size'])
    linear_probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in test_set:
            batch = batch.to(device)
            labels = labels.to(device)
            features = context_enc(batch).mean(dim=1)
            preds = torch.argmax(linear_probe(features), dim=1)
            correct += (preds == labels).sum().item()
            total += batch.size(0)
    accuracy = correct/total 
    print(f"Linear Probe Accuracy: {accuracy:.4f}")




if __name__ == "__main__":
    main()

