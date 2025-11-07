import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

    To write custom datasets, refer to
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.text = f.read()
        
        self.chars = sorted(set(self.text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = 30

    def __len__(self):
        return len(self.data) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        if end_idx >= len(self.data):
            end_idx = len(self.data) - 1
        input_seq = torch.tensor(self.data[start_idx:end_idx-1], dtype=torch.long)
        target_seq = torch.tensor(self.data[start_idx+1:end_idx], dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare('./hw3/data/shakespeare_train.txt')
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        input_seq, target_seq = dataset[i]
        print(f"Input: {''.join([dataset.idx_to_char[idx.item()] for idx in input_seq])}")
        print(f"Target: {''.join([dataset.idx_to_char[idx.item()] for idx in target_seq])}")
