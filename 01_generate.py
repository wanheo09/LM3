# inputs : token id
# outputs: token id
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
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) if prompt else {"input_ids": torch.tensor([[]], dtype=torch.long).to(model.device)}

outputs = model.generate(
    **inputs,
    max_new_tokens=config["max_new_tokens"],
    do_sample=False,
    temperature=None,  # generation_config.json 기본값 무시 (do_sample=False와 충돌 방지)
    top_p=None,        # generation_config.json 기본값 무시 (do_sample=False와 충돌 방지)
    pad_token_id=tokenizer.eos_token_id,  # pad_token 미설정 경고 억제
)

input_ids = inputs["input_ids"][0].tolist()
output_ids = outputs[0].tolist()

print("=== 입력 문장 → token ID ===")
for token_id in input_ids:
    token_str = tokenizer.decode([token_id])
    print(f"  {repr(token_str):20s} → {token_id}")

print("\n=== 출력 token ID → 문장 ===")
for token_id in output_ids:
    token_str = tokenizer.decode([token_id])
    print(f"  {token_id:6d} → {repr(token_str)}")

print("\n=== 최종 출력 문장 ===")
print(tokenizer.decode(output_ids, skip_special_tokens=True))
