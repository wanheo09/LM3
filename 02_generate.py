# inputs : token id
# outputs: token id
#
# 01_generate.py에서는 모델을 블랙박스로 사용했지만,
# 실제 내부 흐름은 아래와 같다:
#
#   token id
#     ↓ embed_tokens  (lookup: vocab_size × hidden_size 행렬)
#   embedding vector  (shape: seq_len × hidden_size)
#     ↓ transformer layers × 32
#   hidden state      (shape: seq_len × hidden_size)  ← embedding과 차원 동일
#     ↓ LM head       (선형 변환: hidden_size → vocab_size)
#   logits            (shape: vocab_size)
#     ↓ argmax        (greedy sampling)
#   token id

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

MODEL = "D:\\models\\hf\\Llama-3.1-8B-Instruct"

# config.json에서 설정을 읽어옴
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")

# config에 지정된 파일에서 프롬프트를 읽어옴
with open(config["prompt_file"], "r", encoding="utf-8") as f:
    prompt = f.read().strip()

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
print(f"입력 token IDs shape: {input_ids.shape}")  # (1, seq_len)

for step in range(config["max_new_tokens"]):
    print(f"\n=== Step {step + 1} ===")

    # 1단계: token ID → embedding vector
    # embed_tokens는 (vocab_size=128256, hidden_size=4096) 크기의 가중치 행렬.
    # 각 token ID를 인덱스로 삼아 해당 행을 조회(lookup)한다.
    input_embeddings = model.model.embed_tokens(input_ids)
    print(f"[1] token IDs → embedding vectors : {tuple(input_ids.shape)} → {tuple(input_embeddings.shape)}")
    #     예) (1, 12) → (1, 12, 4096)

    # 2단계: embedding vector → hidden state  (32개 transformer 레이어 통과)
    # 레이어를 거쳐도 hidden_size=4096 차원은 유지된다.
    # 즉 출력 hidden state의 차원은 입력 embedding과 동일하다.
    transformer_out = model.model(inputs_embeds=input_embeddings)
    hidden_states = transformer_out.last_hidden_state
    print(f"[2] embedding vectors → hidden states: {tuple(input_embeddings.shape)} → {tuple(hidden_states.shape)}")
    #     예) (1, 12, 4096) → (1, 12, 4096)

    # 3단계: hidden state → logits  (LM head)
    # 다음 토큰을 예측하려면 마지막 위치의 hidden state만 필요하다.
    # LM head는 hidden_size(4096) → vocab_size(128256) 선형 변환이다.
    last_hidden = hidden_states[:, -1, :]   # (1, 4096)
    logits      = model.lm_head(last_hidden) # (1, 128256)
    print(f"[3] last hidden state → logits (LM head): {tuple(last_hidden.shape)} → {tuple(logits.shape)}")

    # 4단계: greedy sampling — logits에서 가장 큰 값의 인덱스를 다음 token ID로 선택
    next_token_id  = logits.argmax(dim=-1)           # (1,)
    next_token_str = tokenizer.decode(next_token_id)
    print(f"[4] argmax → next token ID: {next_token_id.item()} → {next_token_str!r}")

    # 선택한 토큰을 입력 끝에 이어붙여 다음 스텝에서 사용
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

print("\n최종 생성 결과:")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
