import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_stl10_dataloader(batch_size, image_size=96):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    dataset = datasets.STL10(root='data', split='unlabeled', download=True, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=0)

def sample_mask(num_patches_side=12, target_blocks = 4, target_scale=(0.15, 0.2), context_scale=0.85):
    tindex = []
    cindex = []
    for _ in range(target_blocks):
        index = []
        maxp = int(target_scale[1] * num_patches_side**2)
        h = torch.randint(1, max(2, int(maxp**0.5)), (1,)).item()
        w = torch.randint(1, max(2, int(maxp**0.5)), (1,)).item()
        startr = torch.randint(0, num_patches_side - h, (1,)).item()
        startc = torch.randint(0, num_patches_side - w, (1,)).item()
        for r in range(h):
            for c in range(w):
                index.append((startr + r)*num_patches_side + (startc + c))
        tindex.append(index)
    tindex = [torch.tensor(block) for block in tindex]
    minc = int(context_scale * num_patches_side**2)
    h = torch.randint(int(minc**0.5), num_patches_side + 1, (1,)).item()
    w = torch.randint(int(minc**0.5), num_patches_side + 1, (1,)).item()
    startr = torch.randint(0, max(1, num_patches_side - h), (1,)).item()
    startc = torch.randint(0, max(1, num_patches_side - w), (1,)).item()
    for r in range(h):
        for c in range(w):
            cindex.append((startr + r)*num_patches_side + (startc + c))
    cindex = torch.tensor([i for i in cindex if i not in torch.cat(tindex)])
    return cindex, tindex
