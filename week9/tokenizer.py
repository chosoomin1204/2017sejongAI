from  nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer 

input_text = "Do you know how tokenization works? It's actually quite easy! Let's analyze a couple of words and figure it out."


print("\nSentence Tokenizer:")
print(sent_tokenize(input_text))

print("\nWord tokenizer : ")
print(word_tokenize(input_text))
