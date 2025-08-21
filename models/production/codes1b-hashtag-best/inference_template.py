### CodeS-1B Italian Hashtag Generator Usage:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("outputs/codes-1b-hashtag-generator")
base_model = AutoModelForCausalLM.from_pretrained("seeklhy/codes-1b", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "outputs/codes-1b-hashtag-generator")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate hashtags from Italian text
def generate_hashtags(text):
    prompt = '''### Task: Generate relevant hashtags for this Italian text
### Text: {text}
### Hashtags: '''.format(text=text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract only generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean result
    hashtags = result.strip().split('\n')[0].split('###')[0].strip()
    
    return hashtags

# Example usage:
# hashtags = generate_hashtags("Bologna, 15 maggio 2025 — L'Università di Bologna inaugura il nuovo centro di ricerca sulla vulcanologia")
# print(hashtags)  # Should output: #bologna #università #vulcanologia #ricerca
