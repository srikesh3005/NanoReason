from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

#Prompt Compression Function
def compress_prompt(prompt):
    stop_words = {"the", "is", "in", "and", "to", "of", "a", "with", "for", "how"}

    words = prompt.split()

    filtered = []
    for word in words:
        if word.lower() not in stop_words:
            filtered.append(word)

    # ADD structure back
    return "Explain: " + " ".join(filtered)

#resaoning controller

def reasoning_controller(prompt):
    length = len(prompt.split())
    if length <5:
        return {
            "max_length":30,
            "compress":False
        }
    elif length < 15:
        return {
            "max_length":60,
            "compress": True
        }
    else:
        return {
            "max_lenght":120,
            "compress":True
        }
    
#load model

model_name = "all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#main pipeline

prompt = "Explain recursion in simple terms"
embedding = model.encode(prompt)
print(len(embedding))
config = reasoning_controller(prompt)

if config["compress"]:
    processed_prompt = compress_prompt(prompt)
else:
    processed_prompt = prompt

inputs = tokenizer(processed_prompt, return_tensors="pt",truncation=True,max_length=50)
outputs = model.generate(
    **inputs,
    max_length=config["max_length"],
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

result = tokenizer.decode(outputs[0])

print("Original Prompt:", prompt)
print("Processed Prompt:", processed_prompt)
print("Generated Response:", result)