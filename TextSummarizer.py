import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('stopwords')

def summarize_text():
    # Get text from input field
    input_text = input_text_box.get("1.0", tk.END)

    # Tokenizing the text 
    stopWords = set(stopwords.words("english")) 
    words = word_tokenize(input_text) 

    # Creating a frequency table to keep the score of each word 
    freqTable = dict() 
    for word in words: 
        word = word.lower() 
        if word in stopWords: 
            continue
        if word in freqTable: 
            freqTable[word] += 1
        else: 
            freqTable[word] = 1

    # Creating a dictionary to keep the score of each sentence 
    sentences = sent_tokenize(input_text) 
    sentenceValue = dict() 

    for sentence in sentences: 
        for word, freq in freqTable.items(): 
            if word in sentence.lower(): 
                if sentence in sentenceValue: 
                    sentenceValue[sentence] += freq 
                else: 
                    sentenceValue[sentence] = freq 

    sumValues = 0
    for sentence in sentenceValue: 
        sumValues += sentenceValue[sentence] 

    # Average value of a sentence from the original text 
    if len(sentenceValue) > 0:
        average = int(sumValues / len(sentenceValue)) 
    else:
        average = 0

    # Storing sentences into our summary. 
    summary = '' 
    for sentence in sentences: 
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)): 
            summary += " " + sentence 

    # Display summary
    output_text_box.delete("1.0", tk.END)
    output_text_box.insert(tk.END, summary)

# Create main window
window = tk.Tk()
window.title("Text Summarizer")

# Create input text box
input_text_box = scrolledtext.ScrolledText(window, width=50, height=10)
input_text_box.grid(column=0, row=0, padx=10, pady=10)

# Create output text box
output_text_box = scrolledtext.ScrolledText(window, width=50, height=10)
output_text_box.grid(column=0, row=1, padx=10, pady=10)

# Create summarize button
summarize_button = tk.Button(window, text="Summarize", command=summarize_text)
summarize_button.grid(column=0, row=2, padx=10, pady=10)

# Start the GUI event loop
window.mainloop()