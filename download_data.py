import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_DIR = "./data"

transform = transforms.Compose([
    transforms.ToTensor(),              # [0,1]
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )                                    # -> [-1,1]
])

train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))


#images, labels = next(iter(train_loader))
#print(images.shape)   # ожидаем: [64, 3, 32, 32]
#print(images.min(), images.max())  # примерно [-1, 1]