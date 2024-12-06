from imports import *
sys.path.append('tacotron2')
from text import text_to_sequence

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, hparams):
        self.data = data
        self.text_cleaners = hparams.text_cleaners

    def __getitem__(self, index):
        mel, text = self.data[index]
        mel = torch.tensor(mel, dtype=torch.float32)
        text = torch.LongTensor(text_to_sequence(text, self.text_cleaners))
        return mel, text

    def __len__(self):
        return len(self.data)
