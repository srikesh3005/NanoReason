import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def compress_prompt(prompt):
    words = word_tokenize(prompt)

    stop_words = set(stopwords.words('english'))

    filtered_words = []
    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)

    compressed_prompt = " ".join(filtered_words)
    return compressed_prompt

prompt = "Explain in detail how recursion works in computer science with examples"

compressed = compress_prompt(prompt)

print("Original:", prompt)
print("Compressed:", compressed)