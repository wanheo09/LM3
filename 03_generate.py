# inputs : token id
# outputs: token id
#
# 02_generate.py에서는 model.model()을 한 번에 호출했지만,
# 여기서는 각 transformer 블록 내부를 직접 순회한다.
#
# LLaMA transformer 블록 구조 (레이어 1개):
#
#   hidden_states
#     ↓ input_layernorm          (RMSNorm — attention 입력 정규화)
#     ↓ self_attn                (Multi-head Attention + RoPE)
#     + residual                 (잔차 연결: 정규화 전 값을 더함)
#     ↓ post_attention_layernorm (RMSNorm — MLP 입력 정규화)
#     ↓ mlp                      (SwiGLU Feed-Forward Network)
#     + residual                 (잔차 연결: 정규화 전 값을 더함)
#   hidden_states
#
# 이 블록이 32회 반복되고, 마지막에 최종 RMSNorm을 한 번 더 적용한다.

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
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"[1] token IDs → embedding vectors: {tuple(input_ids.shape)} → {tuple(hidden_states.shape)}")
    #     예) (1, 12) → (1, 12, 4096)

    # 2단계: 32개 transformer 블록을 직접 순회
    print(f"[2] transformer 블록 × {len(model.model.layers)}개 순회:")

    seq_len = hidden_states.shape[1]

    # RoPE(Rotary Position Embedding) 계산: 각 토큰의 위치 인덱스로 cos/sin 값을 미리 계산
    # position_ids shape: (1, seq_len) — 0, 1, 2, ... seq_len-1
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    # rotary_emb는 (cos, sin) 튜플을 반환하며, self_attn 내부에서 Q/K에 적용된다
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    for i, layer in enumerate(model.model.layers):
        # === Attention 블록 ===

        # residual 저장: attention 이후 더해줄 값
        residual = hidden_states

        # input_layernorm: RMSNorm으로 hidden_states를 정규화
        # RMSNorm은 LayerNorm에서 평균 빼기를 생략한 경량 버전이다
        normed = layer.input_layernorm(hidden_states)

        # self_attn: Multi-head Attention
        # - Q, K, V를 선형 변환으로 생성
        # - position_embeddings(cos, sin)을 사용해 Q, K에 RoPE 적용
        # - Attention score 계산 후 V와 결합하여 출력
        # 반환값: (attn_output, attn_weights, past_key_value)
        attn_output = layer.self_attn(
            normed,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]

        # residual 연결: attention 입력(정규화 전)을 출력에 더함
        # 이로 인해 레이어를 거쳐도 gradient가 안정적으로 흐른다
        hidden_states = residual + attn_output

        # === MLP 블록 ===

        # residual 저장: MLP 이후 더해줄 값
        residual = hidden_states

        # post_attention_layernorm: MLP 입력 정규화
        normed = layer.post_attention_layernorm(hidden_states)

        # mlp: SwiGLU Feed-Forward Network
        # hidden_size(4096) → intermediate_size(14336) → hidden_size(4096)
        # gate_proj와 up_proj 두 경로를 SiLU 게이트로 결합한다
        mlp_output = layer.mlp(normed)

        # residual 연결
        hidden_states = residual + mlp_output

        # 첫 번째와 마지막 레이어만 출력 (32개 전부 출력하면 너무 길다)
        if i == 0 or i == len(model.model.layers) - 1:
            print(f"    레이어 {i:2d}: hidden_states {tuple(hidden_states.shape)}")

    # 모든 레이어 통과 후 최종 RMSNorm
    hidden_states = model.model.norm(hidden_states)
    print(f"    최종 norm 후: {tuple(hidden_states.shape)}")

    # 3단계: hidden state → logits  (LM head)
    # 다음 토큰을 예측하려면 마지막 위치의 hidden state만 필요하다.
    # LM head는 hidden_size(4096) → vocab_size(128256) 선형 변환이다.
    last_hidden = hidden_states[:, -1, :]    # (1, 4096)
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
