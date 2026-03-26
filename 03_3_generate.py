# inputs : token id
# outputs: token id
#
# 03_1_generate.py에서는 RMSNorm을 torch 연산으로 분해했지만,
# 여기서는 MLP(Feed-Forward) 내부를 직접 구현한다.
# (self_attn은 03_2에서 분해했으므로 여기서는 모듈 직접 호출 유지)
#
# LLaMA 3.1-8B MLP 구조 (SwiGLU):
#
#   normed (입력)
#     ↓ gate_proj   (선형 변환,  hidden_size → intermediate_size)
#     ↓ silu        (게이팅 활성화 함수)
#     ↓ up_proj     (선형 변환,  hidden_size → intermediate_size)
#     ↓ element-wise 곱  (gate × up — SwiGLU의 핵심)
#     ↓ down_proj   (선형 변환,  intermediate_size → hidden_size)
#   mlp_output
#
# LLaMA 3.1-8B MLP 파라미터:
#   hidden_size       = 4096
#   intermediate_size = 14336
#
# SwiGLU 수식:
#   mlp(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
#
# SiLU (Sigmoid Linear Unit) 활성화 함수:
#   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
#   - ReLU와 달리 음수 영역에서도 미분값이 존재한다.
#   - GLU(Gated Linear Unit) 구조에서 게이트 역할을 담당한다.
#
# SwiGLU vs 일반 FFN:
#   일반 FFN : relu(x @ W1 + b1) @ W2 + b2
#   SwiGLU   : (silu(x @ Wg) * (x @ Wu)) @ Wd
#   → 게이트(gate_proj)가 정보 흐름을 동적으로 조절하여 표현력을 높인다.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
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
    RMSNorm: x / sqrt(mean(x²) + ε) × weight
    (03_1_generate.py와 동일)
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance.to(torch.float32) + eps)
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
        normed = rms_norm(
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
        )

        # self_attn은 이 파일의 분해 대상이 아니므로 모듈 직접 호출
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

        # --- MLP(SwiGLU) 내부를 직접 구현 ---

        mlp = layer.mlp

        # 1) gate_proj: hidden_size(4096) → intermediate_size(14336)
        #    게이트 역할 — silu를 통과한 후 up_proj와 곱해져 정보 흐름을 조절한다.
        gate = mlp.gate_proj(normed)  # (1, seq, 14336)

        # 2) up_proj: hidden_size(4096) → intermediate_size(14336)
        #    값(value) 역할 — gate의 결과로 얼마나 통과시킬지 결정된다.
        up = mlp.up_proj(normed)      # (1, seq, 14336)

        # 3) SiLU 활성화 후 element-wise 곱 (SwiGLU 핵심 연산)
        #    silu(gate) : 게이트를 활성화하여 [0, 1] 근방의 값으로 스케일
        #    * up       : gate가 열린 만큼만 up 신호를 통과시킴
        activated = F.silu(gate) * up  # (1, seq, 14336)

        # 4) down_proj: intermediate_size(14336) → hidden_size(4096)
        #    확장된 표현을 다시 원래 차원으로 투영
        mlp_output = mlp.down_proj(activated)  # (1, seq, 4096)

        # 첫 번째 레이어에서 MLP 내부 shape 출력
        if i == 0:
            print(f"    [MLP 내부 — 레이어 0]")
            print(f"      normed shape:    {tuple(normed.shape)}")
            print(f"      gate shape:      {tuple(gate.shape)}")
            print(f"      up shape:        {tuple(up.shape)}")
            print(f"      activated shape: {tuple(activated.shape)}")
            print(f"      mlp_output shape:{tuple(mlp_output.shape)}")

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
