#https://www.youtube.com/watch?v=bZcKYiwtw1I Fine tune step by step 
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, pipeline
import torch


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model_id = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             device_type=device)

generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

result = generation_pipeline("Hello, how are you?", max_new_tokens=25)

print(result[0]["generated_text"])

result = generation_pipeline(["Hello, how are you?", "The captial of China is"], 
                             max_new_tokens=25,
                             do_sample=True,
                             top_k=10,
                             top_p=0.95,
                             num_return_sequences=1)

print(result[0]["generated_text"])


input_prompt = ["Hello how are you doing?", "The captial of China is"]
tokenized_input = tokenizer(input_prompt, return_tensors="pt").to(device)
print(tokenized_input["input_ids"].shape)
print(tokenized_input["input_ids"])

tokenizer.batch_decode(tokenized_input["input_ids"])

prompt_template =[
    {"role": "system", "content": "You are a helpful assistant."},  
    {"role": "user", "content": "Hello how are you doing?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
]

tokenizer.pad_token = tokenizer.eos_token

prompt_template = tokenizer.apply_chat_template(prompt_template,
                                                add_generation_prompt=False,
                                                tokenize=True,
                                                padding=True,
                                                continuation=True,
                                                return_tensors="pt")

print(prompt_template)

import pandas as pd
import json

df = pd.read_csv("data/train.csv")
df.head()


text = "Hello how are"

tokenized_text = tokenizer([text], return_tensors="pt")["input_ids"].to(device)

















