import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        if not line:
            return None
        
        encoding = self.tokenizer(
            line,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        return encoding

def collate_fn(batch):
    # 过滤掉空行
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    input_ids = torch.stack([item['input_ids'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

def save_batch_as_json(batch, batch_idx):
    if batch is None:
        return
    
    batch_size = batch['input_ids'].size(0)
    for i in range(batch_size):
        encoding_dict = {
            'input_ids': batch['input_ids'][i].tolist(),
            'token_type_ids': batch['token_type_ids'][i].tolist(),
            'attention_mask': batch['attention_mask'][i].tolist()
        }
        
        output_file = f'output_batch_{batch_idx}_sample_{i}.json'
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(encoding_dict, json_file, ensure_ascii=False, indent=4)

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('../../plm/bert-base-uncased')

# 创建 Dataset 和 DataLoader
file_path = 'input.txt'
dataset = TextDataset(file_path, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# 处理数据并保存为 JSON
for batch_idx, batch in enumerate(dataloader):
    save_batch_as_json(batch, batch_idx)

print("处理完成")
