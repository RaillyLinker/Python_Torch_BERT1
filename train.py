import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Tuple
from bert import BERT, BERTLM  # 기존 정의된 BERT, BERTLM 클래스 사용

# --- 1) 토크나이저 로딩 ---
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# --- 2) 데이터셋 불러오기 & 전처리 ---
dataset = load_dataset('wikipedia', '20220301.en', split='train[:1%]', trust_remote_code=True)


def preprocess_function_with_nsp(examples):
    # 입력 예시: examples["text"]는 리스트
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    nsp_label_list = []

    for text in examples["text"]:
        sentences = text.split(". ")
        if len(sentences) < 2:
            # 문장이 너무 적으면 스킵하거나 빈 값 넣기
            input_ids_list.append([tokenizer.pad_token_id] * 128)
            token_type_ids_list.append([0] * 128)
            attention_mask_list.append([0] * 128)
            nsp_label_list.append(0)
            continue

        # 그냥 첫 두 문장만 사용 (예시)
        first = sentences[0]
        second = sentences[1]
        label = 0  # is next

        encoded = tokenizer(
            first,
            second,
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        input_ids_list.append(encoded["input_ids"])
        token_type_ids_list.append(encoded["token_type_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        nsp_label_list.append(label)

    return {
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "attention_mask": attention_mask_list,
        "nsp_label": nsp_label_list
    }


tokenized_datasets = dataset.map(
    preprocess_function_with_nsp,
    batched=True,
    remove_columns=["text"]
)

# 빠른 테스트를 위한 일부 데이터만 사용
# tokenized_datasets = tokenized_datasets.select(range(100))


# --- 3) Masked Language Modeling 마스킹 함수 ---
def mask_tokens(inputs: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = inputs.clone().long()

    # Special tokens은 mask하지 않음
    probability_matrix = torch.full(labels.shape, 0.15)

    # 한 문장(1D list) 전체를 한번에 넘겨야 함
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        labels.tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Loss 계산에서 무시할 토큰

    inputs = inputs.clone().long()

    # 80%는 [MASK] 토큰으로 대체
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10%는 랜덤 토큰으로 대체
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 나머지 10%는 원래 토큰 유지 (변경 없음)

    return inputs, labels


# --- 4) DataLoader용 Dataset 클래스 ---
class BertPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, tokenizer):
        self.inputs = tokenized_dataset['input_ids']
        self.token_type_ids = tokenized_dataset['token_type_ids']
        self.attention_mask = tokenized_dataset['attention_mask']
        self.nsp_labels = tokenized_dataset['nsp_label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[idx])
        token_type_ids = torch.tensor(self.token_type_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])
        nsp_label = torch.tensor(self.nsp_labels[idx], dtype=torch.long)

        input_ids_masked, labels = mask_tokens(input_ids, self.tokenizer)

        return {
            'input_ids': input_ids_masked,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'mlm_labels': labels,
            'nsp_labels': nsp_label
        }


# --- 5) 학습 루프 ---
if __name__ == "__main__":
    print("train process start")

    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERTLM(BERT(vocab_size=vocab_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_nsp = nn.CrossEntropyLoss()

    train_dataset = BertPretrainDataset(tokenized_datasets, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    scaler = GradScaler()

    model.train()
    for epoch in range(3):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            mlm_labels = batch['mlm_labels'].to(device, non_blocking=True)
            nsp_labels = batch['nsp_labels'].to(device, non_blocking=True)

            with autocast():
                pred_nsp, pred_mlm = model(input_ids, token_type_ids)

                mlm_loss = criterion_mlm(pred_mlm.view(-1, vocab_size), mlm_labels.view(-1))
                nsp_loss = criterion_nsp(pred_nsp, nsp_labels)

                loss = mlm_loss + nsp_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n if pbar.n > 0 else 1))

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model, "model_full.pth")
