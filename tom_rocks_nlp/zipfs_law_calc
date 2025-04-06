import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tkinter import Tk, filedialog

def select_file():
    """Opens a file dialog to select a text file."""
    root = Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(title="Select a text file", filetypes=[("Text Files", "*.txt")])
    return file_path

def preprocess_text(file_path):
    """Reads and preprocesses text: lowercase, remove punctuation, and tokenize."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation & numbers
    words = text.split()
    return words

def analyze_zipfs_law(words, label):
    """Analyzes word frequency and plots Zipf’s Law."""
    word_counts = Counter(words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(sorted_word_counts) + 1)
    frequencies = np.array([count for _, count in sorted_word_counts])
    
    # estimate slope (Zipf's exponent)
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)
    slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
    
    plt.loglog(ranks, frequencies, marker="o", linestyle="none", markersize=3, label=f"{label} (slope={slope:.2f})")
    return slope

# select Latin and English text files
print("Select Latin text file")
latin_file = select_file()
print("Select English text file")
english_file = select_file()

# process texts
latin_words = preprocess_text(latin_file)
english_words = preprocess_text(english_file)

# plot Zipf’s Law
plt.figure(figsize=(8, 5))
latin_slope = analyze_zipfs_law(latin_words, "Latin Text")
english_slope = analyze_zipfs_law(english_words, "English Text")

# plotting stuff
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf's Law in Latin and English Texts")
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated Zipf's Law exponent for Latin text: {latin_slope:.2f}")
print(f"Estimated Zipf's Law exponent for English text: {english_slope:.2f}")

