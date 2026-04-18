import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2")


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


memory = []

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_memory(embedding, threshold=0.8):
    best_score = 0
    best_response = None

    for item in memory:
        score = cosine_sim(embedding, item["embedding"])
        if score > best_score:
            best_score = score
            best_response = item["response"]

    if best_score > threshold:
        return best_response, best_score

    return None, best_score

def store_memory(prompt, embedding, response):
    memory.append({
        "prompt": prompt,
        "embedding": embedding,
        "response": response
    })


def compress_prompt(prompt):
    stop_words = {"the", "is", "in", "and", "to", "of", "a", "with", "for", "how"}

    words = prompt.split()
    filtered = [w for w in words if w.lower() not in stop_words]

    compressed = " ".join(filtered)

    if not compressed.lower().startswith("explain"):
        compressed = "Explain: " + compressed

    return compressed


def reasoning_controller(prompt):
    length = len(prompt.split())

    if length < 5:
        return {"max_length": 40, "compress": False}
    elif length < 15:
        return {"max_length": 80, "compress": True}
    else:
        return {"max_length": 120, "compress": True}


prompt = "What is recursion?"

# Step 1: embedding
embedding = embed_model.encode(prompt)

# Step 2: memory check
response, score = retrieve_memory(embedding)

if response:
    print("Reused from memory (score:", score, ")")
    print(response)

else:
    print("Generating new response...")

    config = reasoning_controller(prompt)

    if config["compress"]:
        processed_prompt = compress_prompt(prompt)
    else:
        processed_prompt = prompt

  
    inputs = tokenizer(processed_prompt, return_tensors="pt", truncation=True, max_length=50)


    outputs = llm_model.generate(
        **inputs,
        max_length=config["max_length"],
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Original:", prompt)
    print("Processed:", processed_prompt)
    print("Response:", result)


    store_memory(prompt, embedding, result)