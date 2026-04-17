from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')

# prompt = "Explain recursion in simple terms"


# embedding = model.encode(prompt)

# print(len(embedding))  

import sklearn


from sklearn.metrics.pairwise import cosine_similarity


# q1 = "explain recursion"
# q2 = "what is recursion"

# emb1 = model.encode(q1)
# emb2 = model.encode(q2)

# score = cosine_similarity([emb1],[emb2])[0][0]

# print(score)

def is_similar(prompt1, prompt2, threshold=0.8):
    emb1 = model.encode(prompt1)
    emb2 = model.encode(prompt2)

    score = cosine_similarity([emb1], [emb2])[0][0]

    return score > threshold
print(is_similar("Explain recursion", "What is recursion?"))
print(is_similar("Explain recursion", "What is football?"))

# def complexity_score(prompt):
#     length = len(prompt.split())

#     if length < 5:
#         return "simple"
#     elif length < 15:
#         return "medium"
#     else:
#         return "complex"
    
# def reasoning_controller(prompt):
#     level = complexity_score(prompt)

#     if level == "simple":
#         return {"max_length": 40, "compress": False}
#     elif level == "medium":
#         return {"max_length": 80, "compress": True}
#     else:
#         return {"max_length": 150, "compress": False}
# print(reasoning_controller(prompt))
