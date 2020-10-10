import fasttext
import csv

model = fasttext.load_model('cc.en.300.bin') # Loading the model
print(f'The dimension of the word vectors is: {model.get_dimension()}') 
# The answer is 300.
# Deleted the model because of its huge size

words = "What are the main symptoms of diabetes?".split() + "Tell me how I may treat bruxism".split() + "What is the cause of a stroke?".split()

# Removing characters other than alphanumeroc from the words
for i, word in enumerate(words):
    if word.find('?'):
        word = word.replace('?', '')
    words[i] = word.lower()

# Reading the word vectors for the unique words since two same words would give the same vector
unique_words = set(words)

with open('unique_words.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(["word, word vector"])
    for word in unique_words:
        write.writerow([word, model.get_word_vector(word)])
    
    

