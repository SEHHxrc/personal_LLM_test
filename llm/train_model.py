from torch.utils.data import Dataset
import json


class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        data = {}
        with open(data_file, 'r') as f:
            for ids, line in enumerate(f):
                sample = json.loads(line.strip())
                data[ids] = sample
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

train_data = AFQMC('afqmc_public/train.json')
valid_data = AFQMC('afqmc_public/dev.json')


