from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        content = self.x[idx]
        label = self.y[idx]
        sample = {'content': content, 'label': label}

        return sample

dataset = TweetDataset(None, None)
