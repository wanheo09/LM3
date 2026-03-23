from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "C:\\Users\\wanhe\\.cache\\huggingface\\hub\\models--meta-llama--Meta-Llama-3-8B\\snapshots\\8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")

prompt = input("Enter prompt (or press Enter to skip): ")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) if prompt else {"input_ids": torch.tensor([[]], dtype=torch.long).to(model.device)}

outputs = model.generate(
    **inputs,
    max_new_tokens=4,
    max_length=2048,
    do_sample=False,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
