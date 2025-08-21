### CodeS-1B Italian Text-to-SQL Usage:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("outputs/codes-1b-text2sql")
base_model = AutoModelForCausalLM.from_pretrained("seeklhy/codes-1b", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "outputs/codes-1b-text2sql")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate SQL from Italian question
def generate_sql(question):
    prompt = '''### Task: Convert this Italian question to SQL query
### Question: {question}
### SQL: '''.format(question=question)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Extract only generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean result
    sql = result.strip()
    if 'SELECT' not in sql.upper():
        return "Error: No valid SQL generated"
    
    return sql

# Example usage:
# sql = generate_sql("Mostra tutti i clienti di Milano")
# print(sql)  # Should output: SELECT * FROM clienti WHERE citta = 'Milano';
