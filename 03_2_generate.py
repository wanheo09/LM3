# inputs : token id
# outputs: token id
#
# 03_1_generate.py에서는 RMSNorm을 torch 연산으로 분해했지만,
# 여기서는 self_attn 내부를 직접 구현한다.
#
# LLaMA 3.1-8B Attention 구조:
#
#   normed (입력)
#     ↓ q_proj / k_proj / v_proj   (선형 변환으로 Q, K, V 생성)
#     ↓ reshape                    (멀티헤드 형태로 분리)
#     ↓ RoPE                       (Q, K에 위치 정보 주입)
#     ↓ GQA repeat                 (K, V를 Q 헤드 수에 맞게 반복 — Grouped Query Attention)
#     ↓ Q @ K^T / sqrt(head_dim)   (Attention score)
#     ↓ causal mask + softmax      (미래 토큰 마스킹 후 확률 분포 변환)
#     ↓ @ V                        (Value 가중합)
#     ↓ o_proj                     (출력 선형 변환)
#   attn_output
#
# LLaMA 3.1-8B Attention 파라미터:
#   hidden_size       = 4096
#   num_heads         = 32   (Q 헤드 수)
#   num_kv_heads      = 8    (K, V 헤드 수 — GQA로 Q보다 적음)
#   head_dim          = 128  (= hidden_size / num_heads = 4096 / 32)
#   num_groups        = 4    (= num_heads / num_kv_heads — 각 KV 헤드가 담당하는 Q 헤드 수)
#
# RoPE (Rotary Position Embedding):
#   - 위치 정보를 더하거나 concat하는 대신, 회전 행렬로 Q/K 벡터에 직접 곱한다.
#   - 수식: q_rot = q * cos + rotate_half(q) * sin
#   - rotate_half: 벡터 후반부의 부호를 반전하고 앞/뒤를 교환
#                  [x1, x2, ..., xn] → [-x(n/2+1), ..., -xn, x1, ..., x(n/2)]
#
# GQA (Grouped Query Attention):
#   - K, V 헤드를 Q보다 적게 두어 메모리와 연산을 절감하는 기법.
#   - 각 KV 헤드 하나가 Q 헤드 4개를 담당한다.
#   - 구현: K, V를 num_groups(=4)배 repeat_interleave하여 Q와 크기를 맞춤.
#
# Scaled Dot-Product Attention:
#   Attention(Q, K, V) = softmax( Q @ K^T / sqrt(head_dim) + mask ) @ V

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
    RMSNorm: x / sqrt(mean(x²) + ε) × weight
    (03_1_generate.py와 동일)
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(variance.to(torch.float32) + eps)
    return (weight * x_normed).to(x.dtype)


def rotate_half(x):
    """
    RoPE에서 사용하는 벡터 회전 헬퍼.

    head_dim(128)을 절반으로 나눠 앞/뒤 블록을 교환하고 뒤 블록 부호를 반전한다.
      입력: [x1 | x2]  (각 블록 크기 = head_dim / 2 = 64)
      출력: [-x2 | x1]

    이 연산과 cos/sin을 결합하면 복소수 회전과 수학적으로 동일한 효과를 낸다.
    """
    x1 = x[..., : x.shape[-1] // 2]   # 앞쪽 절반
    x2 = x[..., x.shape[-1] // 2 :]   # 뒤쪽 절반
    return torch.cat((-x2, x1), dim=-1)


for step in range(config["max_new_tokens"]):
    print(f"\n=== Step {step + 1} ===")

    # 1단계: token ID → embedding vector
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"[1] token IDs → embedding vectors: {tuple(input_ids.shape)} → {tuple(hidden_states.shape)}")

    # 2단계: 32개 transformer 블록을 직접 순회
    print(f"[2] transformer 블록 × {len(model.model.layers)}개 순회:")

    seq_len = hidden_states.shape[1]

    # RoPE cos/sin 값을 미리 계산
    # rotary_emb 는 (cos, sin) 튜플을 반환한다. shape: (1, seq_len, head_dim)
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    cos, sin = position_embeddings  # 각각 (1, seq_len, 128)

    # cos/sin에 헤드 차원(dim=1)을 추가: (1, 1, seq_len, 128)
    # 이후 Q (1, 32, seq, 128) / K (1, 8, seq, 128) 과 브로드캐스트로 곱해진다.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    for i, layer in enumerate(model.model.layers):
        # === Attention 블록 ===

        residual = hidden_states

        # input_layernorm을 torch 연산으로 직접 수행 (03_1과 동일)
        normed = rms_norm(
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
        )

        # --- self_attn 내부를 직접 구현 ---

        attn = layer.self_attn
        num_heads    = attn.config.num_attention_heads   # 32
        num_kv_heads = attn.config.num_key_value_heads   # 8
        head_dim     = attn.head_dim                     # 128
        num_groups   = num_heads // num_kv_heads # 4
        bsz          = normed.shape[0]           # 1 (batch size)

        # 1) Q, K, V 선형 투영
        #    q_proj: (hidden_size=4096) → (num_heads    × head_dim = 4096)
        #    k_proj: (hidden_size=4096) → (num_kv_heads × head_dim = 1024)
        #    v_proj: (hidden_size=4096) → (num_kv_heads × head_dim = 1024)
        q = attn.q_proj(normed)  # (1, seq, 4096)
        k = attn.k_proj(normed)  # (1, seq, 1024)
        v = attn.v_proj(normed)  # (1, seq, 1024)

        # 2) 멀티헤드 reshape: (batch, seq, heads × head_dim) → (batch, heads, seq, head_dim)
        q = q.view(bsz, seq_len, num_heads,    head_dim).transpose(1, 2)  # (1, 32, seq, 128)
        k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (1,  8, seq, 128)
        v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # (1,  8, seq, 128)

        # 3) RoPE 적용: Q, K에 위치 정보를 회전으로 주입
        #    수식: x_rot = x * cos + rotate_half(x) * sin
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        # 4) GQA: K, V를 Q 헤드 수에 맞게 반복
        #    8개 KV 헤드를 각각 4번씩 복제 → 32개로 확장
        #    (1, 8, seq, 128) → (1, 32, seq, 128)
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

        # 5) Attention score: Q @ K^T / sqrt(head_dim)
        #    (1, 32, seq, 128) @ (1, 32, 128, seq) → (1, 32, seq, seq)
        scale  = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 6) Causal mask: 미래 토큰(j > i)의 score를 -inf로 마스킹
        #    torch.triu(..., diagonal=1)은 주 대각선 위쪽(미래)만 추출한다.
        #    softmax 후 해당 위치의 attention weight는 0이 된다.
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"),
            device=scores.device, dtype=scores.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)  # (seq, seq)
        scores = scores + causal_mask  # 브로드캐스트: (1, 32, seq, seq)

        # 7) Softmax: score → attention weight (확률 분포)
        #    각 query 위치(dim=-1)에 대해 key 위치들의 weight 합 = 1
        attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        # (1, 32, seq, seq)

        # 8) V 가중합: attention weight로 Value를 합산
        #    (1, 32, seq, seq) @ (1, 32, seq, 128) → (1, 32, seq, 128)
        attn_output = torch.matmul(attn_weights, v)

        # 9) 헤드 차원 병합 후 출력 투영
        #    (1, 32, seq, 128) → (1, seq, 4096) → o_proj → (1, seq, 4096)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
        attn_output = attn.o_proj(attn_output)

        # 첫 번째 레이어에서 attention 내부 shape 출력
        if i == 0:
            print(f"    [attention 내부 — 레이어 0]")
            print(f"      Q shape (RoPE 후):   {tuple(q.shape)}")
            print(f"      K shape (GQA 후):    {tuple(k.shape)}")
            print(f"      V shape (GQA 후):    {tuple(v.shape)}")
            print(f"      scores shape:        {tuple(scores.shape)}")
            print(f"      attn_weights shape:  {tuple(attn_weights.shape)}")
            print(f"      attn_output shape:   {tuple(attn_output.shape)}")

        hidden_states = residual + attn_output

        # === MLP 블록 ===

        residual = hidden_states

        # post_attention_layernorm을 torch 연산으로 직접 수행
        normed = rms_norm(
            hidden_states,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )

        # mlp는 이 파일의 분해 대상이 아니므로 모듈 직접 호출
        mlp_output = layer.mlp(normed)

        hidden_states = residual + mlp_output

        if i == 0 or i == len(model.model.layers) - 1:
            print(f"    레이어 {i:2d}: hidden_states {tuple(hidden_states.shape)}")

    # 최종 RMSNorm (model.norm)
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
