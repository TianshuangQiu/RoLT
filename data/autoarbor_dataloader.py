from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BASE_DIR = "/home/sandeepmukh/autoarborist"

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(
            #     brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            # ),
            transforms.ToTensor(),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Normalize((0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)),
            transforms.RandomErasing(p=0.5),
            transforms.Resize((224, 224)),  # , antialias=True),  # type: ignore
            # transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4266, 0.4126, 0.3965), (0.2399, 0.2279, 0.2207)),
            transforms.Resize((224, 224)),  # antialias=True),  # type: ignore
            transforms.ToTensor(),
        ]
    ),
}

my_datasets = {
    s: datasets.ImageFolder(f"{BASE_DIR}/data/aerial_images/{s}/", data_transforms[s])
    for s in ["train", "val"]
}


def get_dataloader(batch_size=256, num_workers=4):
    dataloaders = {
        s: DataLoader(
            my_datasets[s],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=(s == "train"),
        )
        for s in ["train", "val"]
    }
    return dataloaders
