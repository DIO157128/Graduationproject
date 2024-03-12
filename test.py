import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def custom_collate(batch):
    batch_size = len(batch)
    batch_samples = [batch[i][0] for i in range(batch_size)]
    batch_labels = [batch[i][1] for i in range(batch_size)]
    return batch_samples, batch_labels


# Example data
data = [
    ([[1, 2, 3], [4, 5], [6]], 10),
    ([[7, 8, 9, 10], [11, 12, 13], [14, 15]], 20),
    ([[16, 17, 18], [19, 20, 21, 22], [23, 24, 25, 26, 27]], 30),
    ([[28, 29], [30, 31, 32, 33], [34, 35, 36]], 40),
    ([[37, 38, 39, 40], [41, 42], [43, 44, 45, 46, 47, 48]], 50)
]


# Create custom dataset and dataloader
custom_dataset = CustomDataset(data)
sampler = RandomSampler(custom_dataset)
custom_dataloader = DataLoader(custom_dataset, sampler=sampler,batch_size=2, collate_fn=custom_collate)

# Iterate over batches
for batch_samples, batch_labels in custom_dataloader:
    print("Batch samples:")
    for sample in batch_samples:
        print(sample)
    print("Batch labels:", batch_labels)


