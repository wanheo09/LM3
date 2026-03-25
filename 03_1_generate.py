# inputs : token id
# outputs: token id
#
# 03_generate.py에서는 layer.input_layernorm / layer.post_attention_layernorm /
# model.model.norm 을 직접 호출했지만,
# 여기서는 RMSNorm 연산을 torch 기본 연산으로 직접 풀어서 보여준다.
#
# RMSNorm 수식:
#
#   rms   = sqrt( mean(x²) + ε )
#   출력  = (x / rms) × w
#
#   - x : 입력 텐서  (hidden_states)
#   - ε : 수치 안정성을 위한 작은 값 (variance_epsilon, 보통 1e-5)
#   - w : 학습된 스케일 파라미터 (weight, shape: hidden_size)
#
# LayerNorm과의 차이:
#   LayerNorm  : 평균을 빼고(μ) 분산으로 나눔  →  (x - μ) / std × w + b
#   RMSNorm    : 평균을 빼지 않고 RMS로만 나눔  →  x / rms × w
#   → 계산이 더 가볍고 LLaMA 계열에서 표준으로 사용됨

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# config.json에서 설정을 읽어옴
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL = config["model_path"]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")

# config에 지정된 파일에서 프롬프트를 읽어옴
with open(config["prompt_file"], "r", encoding="utf-8") as f:
    prompt = f.read().strip()

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
print(f"입력 token IDs shape: {input_ids.shape}")  # (1, seq_len)


def rms_norm(x, weight, eps):
    """
    RMSNorm을 torch 기본 연산으로 직접 구현한다.

    x      : 입력 텐서,          shape (..., hidden_size)
    weight : 학습된 스케일 벡터, shape (hidden_size,)
    eps    : 수치 안정성 항,     LlamaRMSNorm.variance_epsilon 값 사용

    단계:
      1) x²의 평균을 마지막 차원(hidden_size)에 걸쳐 계산   → variance
      2) sqrt(variance + ε) 의 역수(rsqrt)를 곱해 정규화    → x_normed
      3) 학습된 weight로 스케일                              → 최종 출력
    """
    # 1단계: 각 토큰 위치별 RMS 계산
    #   x.pow(2)          : 원소별 제곱
    #   .mean(-1, keepdim=True) : hidden_size 차원 평균 → shape (..., 1)
    variance = x.pow(2).mean(-1, keepdim=True)

    # 2단계: (variance + ε)^(-0.5) 를 곱해 정규화
    #   torch.rsqrt(v) = 1 / sqrt(v)  — sqrt + 나눗셈을 한 번에 처리
    #   x.to(torch.float32): bfloat16은 정밀도가 낮아 float32로 올려서 계산
    x_normed = x.to(torch.float32) * torch.rsqrt(variance.to(torch.float32) + eps)

    # 3단계: weight 스케일 적용 후 원래 dtype으로 복원
    return (weight * x_normed).to(x.dtype)


for step in range(config["max_new_tokens"]):
    print(f"\n=== Step {step + 1} ===")

    # 1단계: token ID → embedding vector
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"[1] token IDs → embedding vectors: {tuple(input_ids.shape)} → {tuple(hidden_states.shape)}")

    # 2단계: 32개 transformer 블록을 직접 순회
    print(f"[2] transformer 블록 × {len(model.model.layers)}개 순회:")

    seq_len = hidden_states.shape[1]

    # RoPE cos/sin 값을 미리 계산
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    for i, layer in enumerate(model.model.layers):
        # === Attention 블록 ===

        residual = hidden_states

        # input_layernorm을 torch 연산으로 직접 수행
        #   layer.input_layernorm.weight         : 학습된 스케일 벡터
        #   layer.input_layernorm.variance_epsilon: ε 값
        normed = rms_norm(
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
        )

        attn_output = layer.self_attn(
            normed,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]

        hidden_states = residual + attn_output

        # === MLP 블록 ===

        residual = hidden_states

        # post_attention_layernorm을 torch 연산으로 직접 수행
        normed = rms_norm(
            hidden_states,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )

        mlp_output = layer.mlp(normed)

        hidden_states = residual + mlp_output

        if i == 0 or i == len(model.model.layers) - 1:
            print(f"    레이어 {i:2d}: hidden_states {tuple(hidden_states.shape)}")

    # 모든 레이어 통과 후 최종 RMSNorm (model.norm) 도 torch 연산으로 직접 수행
    hidden_states = rms_norm(
        hidden_states,
        model.model.norm.weight,
        model.model.norm.variance_epsilon,
    )
    print(f"    최종 norm 후: {tuple(hidden_states.shape)}")

    # 3단계: hidden state → logits  (LM head)
    last_hidden = hidden_states[:, -1, :]    # (1, 4096)
    logits      = model.lm_head(last_hidden) # (1, 128256)
    print(f"[3] last hidden state → logits (LM head): {tuple(last_hidden.shape)} → {tuple(logits.shape)}")

    # 4단계: greedy sampling
    next_token_id  = logits.argmax(dim=-1)
    next_token_str = tokenizer.decode(next_token_id)
    print(f"[4] argmax → next token ID: {next_token_id.item()} → {next_token_str!r}")

    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

print("\n최종 생성 결과:")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
