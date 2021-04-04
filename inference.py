import argparse
import json
import pandas as pd
import torch
from transformers import set_seed, AutoTokenizer, AutoModelForSeq2SeqLM
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

# parser = argparse.ArgumentParser()
# parser.add_argument('--text', type=str, required=True)
# parser.add_argument('--device', type=str, default='cpu')

SEED = 42
set_seed(SEED)

def inference(model, tokenizer, test_df, device):
    model.to(device)
    model.eval()
    results = []
    
    with torch.no_grad():
        for text, gd in zip(test_df['text'], test_df['tag']):
            inputs = tokenizer([text[:1024]], return_tensors='pt')
            del inputs['token_type_ids']
            res = model.generate(**inputs, do_sample=True, num_return_sequences=10)
            generated_summary = list(set([tokenizer.decode(r, skip_special_tokens=True) for r in res]))
            generated = {"text":text, "golden tag": gd, "generated tag": generated_summary}
            results.append(generated)
    return results

if __name__ == '__main__':
#     args = parser.parse_args()
    
    
#     text = args.text
#     device = args.device

    with open('data/Brunch_accm_20210328_test.json', 'r') as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data)
    
#     test_df = df['context']
    device = 'cpu'
    tokenizer = get_kobart_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained("model_checkpoint/checkpoint_20210328_large/saved_checkpoint_5")
#     model.to('cuda')
    
    res = inference(model, tokenizer, test_df, device)
    print(res)
    with open('test_result.json', 'w') as f:
        json.dump(res, f)