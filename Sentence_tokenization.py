import nltk

def sentence_tokenize(text):
    return nltk.sent_tokenize(text)

text = "Hello! How are you? I hope you're doing well. This is a tokenization example."
sentences = sentence_tokenize(text)

print("Sentences:", sentences)
