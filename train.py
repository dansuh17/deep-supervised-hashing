"""
Liu et al., "Deep Supervised Hashing for Fast Image Retrieval"

Train DSH on MNIST
1. MNIST Pair generation (later make this online sampling)
2. Identify hyperparameters, implement loss function
3. Train - log any important information
4. Test image retrieval from Hamming space
5. do a t-SNE on Hamming space
"""
import random
import torch
from torch import optim
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import LiuDSH

# hyperparameters
DATA_ROOT = 'data_out'
LR_INIT = 3e-4
BATCH_SIZE = 64
EPOCH = 40
NUM_WORKERS = 8
CODE_SIZE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, ), std=(0.3081, )),
])


class MNISTPairDataset(Dataset):
    def __init__(self, data_root: str, transform=None, train=True):
        super().__init__()
        self.dataset = MNIST(root=data_root, train=train, transform=transform, download=True)
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # return image pair
        x_img, x_target = self.dataset[item]
        pair_idx = item
        # choose a different index
        while pair_idx == item:
            pair_idx = random.randint(0, self.size)

        y_img, y_target = self.dataset[pair_idx]
        target_equals = 0 if x_target == y_target else 1
        return x_img, y_img, target_equals


train_pair_dataset = MNISTPairDataset(data_root=DATA_ROOT, train=True, transform=mnist_transform)
test_pair_dataset = MNISTPairDataset(data_root=DATA_ROOT, train=False, transform=mnist_transform)
train_dataloader = DataLoader(
    train_pair_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS)

model = LiuDSH(code_size=CODE_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR_INIT)

if __name__ == '__main__':
    writer = SummaryWriter()
    train = True

    for e in range(EPOCH):
        for x_imgs, y_imgs, target_equals in train_dataloader:
            x_out = model(x_imgs)
            print(x_out.size())
            print(target_equals)

            # T1: 0.5 * (1 - y) * dist(x1, x2)
            # T2: 0.5 * y * max(margin - dist(x1, x2), 0)
            # T3: alpha(abs(x1 - 1) + abs(x2 - 1))
            loss = 0.5 * (1 - target_equals)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    writer.close()
