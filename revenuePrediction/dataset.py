import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class CustomDataset(Dataset):
    def __init__(self, anot, config):
        super().__init__()
        self.tokenizer = tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.anot = anot
        self.max_seq_len = config["text"]["max_seq_len"]
        
    def __len__(self):
        return len(self.anot)

    def __getitem__(self, idx):
        sample = self.anot.iloc[idx]

        text = self.tokenizer(sample.concatenated, padding='max_length', truncation=True, max_length=self.max_seq_len)
        
#         print(np.shape(text["input_ids"]))
#         print(np.shape(text["attention_mask"]))
        
        return {
            "text": torch.LongTensor(text["input_ids"]),
            "text_mask": torch.LongTensor(text["attention_mask"]),
            "cou": torch.LongTensor([sample.country_of_origin]),
            "lan": torch.LongTensor([sample.languages]),
            "sta": torch.LongTensor([sample.startYear]),
            "crew": torch.LongTensor([sample.Director]),
            "wri": torch.LongTensor([sample.Writers]),
            "gen": torch.LongTensor(sample.genres),
            "gen_mask": torch.LongTensor(sample.genres_mask),
            "com": torch.LongTensor(sample.production_companies),
            "com_mask": torch.LongTensor(sample.company_mask),
            "cast": torch.LongTensor(sample.cast),
            "cast_mask": torch.LongTensor(sample.cast_mask),
            "bud": torch.FloatTensor([sample.budget]),
            "run": torch.FloatTensor([sample.runtimeMinutes]),
            "rev": torch.FloatTensor([sample.revenue]),
            "idx": torch.LongTensor([idx])
        }