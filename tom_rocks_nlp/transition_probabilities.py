# === Imports ===

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk import bigrams
from tkinter import Tk
from tkinter.filedialog import askopenfilename

nltk.download('punkt')


# === Text Processing ===

def process_text(text):
    """ Preprocess the text: Tokenize and clean (convert to lowercase) """
    tokens = nltk.word_tokenize(text.lower())                  # tokenize and lower case
    tokens = [word for word in tokens if word.isalpha()]       # remove non-alphabetic tokens
    return tokens


def calculate_bigrams(tokens):
    """ Calculate bigrams and their counts """
    bigram_list = list(bigrams(tokens))                        # get all bigrams from token list
    bigram_counts = Counter(bigram_list)                       # count number of each bigram
    return bigram_counts


def calculate_transition_probabilities_with_smoothing(bigram_counts, word_counts, vocabulary_size):
    """ Calculate transition probabilities with Laplace smoothing """
    transition_probs = {}
    for (word1, word2), count in bigram_counts.items():
        prob = (count + 1) / (word_counts[word1] + vocabulary_size)  # smoothing
        transition_probs[(word1, word2)] = prob  
    return transition_probs


# === File Handling ===

def load_corpus():
    """ Load a text corpus using a dynamic file dialog """
    Tk().withdraw()  # Hide the root Tkinter window
    filename = askopenfilename(title="Select a text file", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if filename:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print("No file selected!")
        return None


# === Visualization ===

def plot_transition_probabilities_comparison(english_probs, latin_probs, language_names, smoothing=False):
    """ Visualize transition probabilities for English and Latin on one plot """
    
    # Sort by probability values
    sorted_english_probs = sorted(english_probs.items(), key=lambda x: x[1], reverse=True)
    sorted_latin_probs   = sorted(latin_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N bigrams for each language
    top_n = 1000
    top_english_bigrams = sorted_english_probs[:top_n]
    top_latin_bigrams   = sorted_latin_probs[:top_n]
    
    # Create plot
    plt.figure(figsize=(12, 8))

    labels_english = [' '.join(bigram) for bigram, _ in top_english_bigrams]
    probs_english  = [prob for _, prob in top_english_bigrams]

    labels_latin = [' '.join(bigram) for bigram, _ in top_latin_bigrams]
    probs_latin  = [prob for _, prob in top_latin_bigrams]

    if smoothing:
        # Apply logarithmic transformation for Laplace Smoothed probabilities
        prob_values_english = np.log([p if p > 0 else 1e-10 for p in probs_english])  # Avoid log(0)
        prob_values_latin   = np.log([p if p > 0 else 1e-10 for p in probs_latin])
        plt.title(f'Top {top_n} Bigram Transition Probabilities (Log Scale) - Comparison')
    else:
        prob_values_english = probs_english
        prob_values_latin   = probs_latin
        plt.title(f'Top {top_n} Bigram Transition Probabilities - Comparison')

    # Plot both English and Latin
    plt.barh(labels_english, prob_values_english, color='blue', alpha=0.7, label='English')
    plt.barh(labels_latin, prob_values_latin, color='green', alpha=0.7, label='Latin')

    plt.xlabel('Transition Probability (Log Scale if Smoothing Applied)')
    plt.legend(loc='upper right')
    plt.show()


# === Main Execution ===

# Load English and Latin corpora dynamically
print("Select the English text file:")
english_text = load_corpus()  # Load the English text file
if not english_text:
    raise ValueError("No English text file selected. Exiting.")

print("Select the Latin text file:")
latin_text = load_corpus()  # Load the Latin text file
if not latin_text:
    raise ValueError("No Latin text file selected. Exiting.")

# Process the text data
english_tokens = process_text(english_text)
latin_tokens   = process_text(latin_text)

# Calculate word counts (for denominator in transition probability)
english_word_counts = Counter(english_tokens)
latin_word_counts   = Counter(latin_tokens)

# Calculate bigrams and transition probabilities
english_bigrams = calculate_bigrams(english_tokens)
latin_bigrams   = calculate_bigrams(latin_tokens)

# Calculate Laplace smoothed transition probabilities
english_vocabulary_size = len(set(english_tokens))  # Number of unique words in English corpus
latin_vocabulary_size   = len(set(latin_tokens))    # Number of unique words in Latin corpus

english_transition_probs_smoothed = calculate_transition_probabilities_with_smoothing(
    english_bigrams, english_word_counts, english_vocabulary_size
)

latin_transition_probs_smoothed = calculate_transition_probabilities_with_smoothing(
    latin_bigrams, latin_word_counts, latin_vocabulary_size
)

# Visualize the results on one graph: Log scale and Laplace smoothing
plot_transition_probabilities_comparison(
    english_transition_probs_smoothed,
    latin_transition_probs_smoothed,
    ['English', 'Latin'],
    smoothing=True
)
