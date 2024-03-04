from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv('HF_TOKEN')

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "¿Cómo es Chile?"

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
outputs = tokenizer.batch_decode(generated_ids)

for x in outputs:
    print(x)