import random
import nltk
import requests
from nltk.corpus import wordnet

nltk.download('wordnet')

# Define functions for text augmentation
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Replace underscores with spaces
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(sentence, n):
    words = sentence.split()
    if len(words) <= 2:
        return sentence  # No replacement if the sentence is too short

    new_words = words.copy()
    random_word_list = list(set(words[1:-1]))  # Exclude the first and last words
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def paraphrase(sentence, n):
    words = sentence.split()
    if len(words) <= 2:
        return sentence  # No paraphrasing if the sentence is too short

    new_words = words.copy()
    random_word_list = list(set(words[1:-1]))  # Exclude the first and last words
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    # Simple structure change: reverse the middle part of the sentence
    if len(new_words) > 2:
        new_words = [new_words[0]] + new_words[1:-1][::-1] + [new_words[-1]]

    return ' '.join(new_words)

def random_insertion(sentence, n):
    words = sentence.split()
    for _ in range(n):
        add_word(words)
    return ' '.join(words)

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = words[random.randint(0, len(words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(words)-1)
    words.insert(random_idx, random_synonym)

def random_swap(sentence, n):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    for _ in range(n):
        words = swap_word(words)
    return ' '.join(words)

def swap_word(words):
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

def random_deletion(sentence, p):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)
    if len(new_words) == 0:
        return words[random.randint(0, len(words)-1)]
    return ' '.join(new_words)

# Fetch the dataset
url = "<link_to_your_github_dataset>"
response = requests.get(url)
data = response.text.strip().split("\n")

# Process the dataset
dataset = [tuple(line.split(';')) for line in data]

# Augment the dataset
augmented_dataset = []
original_count = len(dataset)

for modern, shakespearean in dataset:
    augmented_dataset.append((modern, shakespearean))
    augmented_dataset.append((synonym_replacement(modern, 1), shakespearean))
    augmented_dataset.append((paraphrase(modern, 1), shakespearean))
    augmented_dataset.append((random_insertion(modern, 1), shakespearean))
    augmented_dataset.append((random_swap(modern, 1), shakespearean))
    augmented_dataset.append((random_deletion(modern, 0.1), shakespearean))

# Remove duplicates
augmented_dataset = list(set(augmented_dataset))

# Shuffle the augmented dataset
random.shuffle(augmented_dataset)

# Save the augmented dataset to a text file
output_file = 'augmented_shakespeare_style_transfer.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for modern, shakespearean in augmented_dataset:
        f.write(f'{modern};{shakespearean}\n')

# Print the number of new samples generated
new_samples_count = len(augmented_dataset) - original_count
print(f'Number of new samples generated: {new_samples_count}')
print(f'Augmented dataset saved to {output_file}')
