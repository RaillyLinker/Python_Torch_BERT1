import torch
from transformers import BertTokenizerFast


def mask_tokens_for_test(inputs: torch.Tensor, tokenizer):
    """
    테스트용으로 입력 문장에 일부 토큰을 마스킹.
    학습 때처럼 15% 확률로 마스킹. (더 간단하게 하려면 수동 마스크도 가능)
    """
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, 0.15)
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        inputs.tolist(), already_has_special_tokens=True
    )
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return inputs, masked_indices


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 토크나이저 로드
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 2) 저장된 모델 불러오기
    model = torch.load("model_full.pth")
    model.to(device)
    model.eval()

    # 3) 테스트용 문장 입력 (두 문장)
    first_sentence = "The quick brown fox"
    second_sentence = "jumps over the lazy dog"

    # 4) 토크나이저로 인코딩 (max_length=128, padding 등 학습과 동일하게)
    encoded = tokenizer(
        first_sentence,
        second_sentence,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
    token_type_ids = encoded["token_type_ids"].squeeze(0)

    # 5) MLM을 위해 일부 토큰 마스킹
    input_ids_masked, masked_indices = mask_tokens_for_test(input_ids, tokenizer)

    # 6) 배치 차원 추가, GPU로 이동
    input_ids_masked = input_ids_masked.unsqueeze(0).to(device)
    token_type_ids = token_type_ids.unsqueeze(0).to(device)

    # 7) 모델 추론
    with torch.no_grad():
        pred_nsp, pred_mlm = model(input_ids_masked, token_type_ids)
        # pred_nsp: (batch_size=1, 2) NSP 결과 확률 logits
        # pred_mlm: (batch_size=1, seq_len, vocab_size) MLM 예측 logits

    # 8) NSP 예측 결과 해석
    nsp_probs = torch.softmax(pred_nsp, dim=-1)
    is_next_prob = nsp_probs[0, 0].item()
    not_next_prob = nsp_probs[0, 1].item()
    print(f"NSP prediction - is_next: {is_next_prob:.4f}, not_next: {not_next_prob:.4f}")

    # 9) MLM 예측 결과에서 마스킹된 토큰 위치만 출력해보기
    masked_positions = masked_indices.nonzero(as_tuple=False).squeeze(-1).tolist()
    predicted_tokens = torch.argmax(pred_mlm, dim=-1).squeeze(0)  # (seq_len,)

    print("MLM predictions on masked tokens:")
    for pos in masked_positions:
        true_token_id = input_ids[pos].item()
        pred_token_id = predicted_tokens[pos].item()
        true_token = tokenizer.convert_ids_to_tokens(true_token_id)
        pred_token = tokenizer.convert_ids_to_tokens(pred_token_id)
        print(f"Position {pos}: True token = '{true_token}', Predicted token = '{pred_token}'")
