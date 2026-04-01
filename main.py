from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time, in a land far, far away, there lived a"

inputs = tokenizer(prompt,return_tensors="pt")
outputs  = model.generate(**inputs,max_length=100)

print(tokenizer.decode(outputs[0]))

