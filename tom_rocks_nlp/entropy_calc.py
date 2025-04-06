import os
import re
from collections import Counter
import math
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# get full path of the text files directory
directory = os.path.join(os.getcwd(), "text_files")  # adjust if needed

# Check if directory exists
if not os.path.exists(directory):
    raise FileNotFoundError(f"Directory not found: {directory}")

# dynamically select text files
def get_text_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".txt")]

# load text from a file
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

# preprocess text 
def preprocess_text(text):
    text = text.lower() # make lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text

# calculate entropy
def calculate_entropy(text):
    words = word_tokenize(preprocess_text(text))
    total_words = len(words)
    word_counts = Counter(words)

    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    return entropy

# ensure texts are of equal length
def match_text_length(text1, text2):
    words1, words2 = text1.split(), text2.split()
    min_length = min(len(words1), len(words2))
    return " ".join(words1[:min_length]), " ".join(words2[:min_length])

# get text files
text_files = get_text_files(directory)

if len(text_files) < 2:
    raise ValueError(f"Not enough text files found in the directory: {directory}")

# load texts dynamically
latin_text = load_text(text_files[0])
english_text = load_text(text_files[1])

# match text lengths (to make fair)
latin_text, english_text = match_text_length(latin_text, english_text)

# compute entropy for both texts
latin_entropy = calculate_entropy(latin_text)
english_entropy = calculate_entropy(english_text)

# print result of calc
print(f"Latin Entropy: {latin_entropy}")
print(f"English Entropy: {english_entropy}")
