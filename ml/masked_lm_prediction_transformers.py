from datasets import load_dataset
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
import torch


def main():
    # load data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
    text = dataset[10]['text']
    # print(text)

    # load pretrained model
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

    # tokenize text
    text = tokenizer.tokenize(text)
    print(f'Tokenized text: {text}')

    # mask a token
    mask_idx = 5
    text[mask_idx] = tokenizer.mask_token
    print(f'Masked tokens: {text}')

    # masked LM
    input_ids = tokenizer.convert_tokens_to_ids(text)
    # print(input_ids)
    batch = torch.tensor([input_ids])

    model.eval()
    with torch.no_grad():
        output = model(batch, return_dict=True)

    # logit -> token id -> string
    pred = torch.argmax(output.logits[0], dim=-1)[mask_idx].item()
    pred = tokenizer.convert_ids_to_tokens(pred)
    print(pred)


if __name__ == '__main__':
    main()
