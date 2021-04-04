import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SummaryDataset(Dataset):
    def __init__(self, context, summary, tok, enc_max_len, dec_max_len, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len
        self.context = context
        self.summary = summary
        self.pad_index = tok.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([self.pad_index] *(max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs

    def add_ignored_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([self.ignore_index] *(max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs
    
    def __getitem__(self, idx):
        context = self.context[idx]
        summary = self.summary[idx]
        input_ids = self.tok.encode(context)
        input_ids = self.add_padding_data(input_ids, self.enc_max_len)

        label_ids = self.tok.encode(summary, add_special_tokens=False)
        label_ids.append(self.tok.eos_token_id)
        dec_input_ids = [self.tok.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids, self.dec_max_len)
        label_ids = self.add_ignored_data(label_ids, self.dec_max_len)

#         return (torch.tensor(input_ids),
#                 torch.tensor(dec_input_ids),
#                 torch.tensor(label_ids))
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return len(self.context)
    
class SummaryBatchGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
#         print(batch)
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        decoder_input_ids = torch.tensor([item['decoder_input_ids'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).int()
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).int()
        
        return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask}

def get_dataloader(dataset, batch_generator, batch_size=16, shuffle=True):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=batch_generator,
                              num_workers=4)
    return data_loader
