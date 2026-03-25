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
    max_length=2048,
    do_sample=False,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
