
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk
from tkinter import messagebox

# Download stopwords once
nltk.download('stopwords')

# ------------------ Training Section ------------------
texts = [
    "I love this product", "This is amazing", "So happy with the service", "Absolutely wonderful!", 
    "Best purchase ever", "Totally recommend it", "I'm so satisfied", "Fantastic experience", 
    "I would buy this again", "The quality is great", "Excellent customer service",
    "Great value for money", "I'm impressed", "Very enjoyable", "I can't stop smiling", 
    "I feel great about this purchase", "Worth every penny", "I love it so much", 
    "Perfect in every way", "Superb quality", "Outstanding product", "Totally worth it",
    
    "I hate this", "Worst experience", "I'm not happy with this", "Completely broken", 
    "Terrible product", "I want a refund", "Very disappointed", "The worst purchase I made",
    "It doesn't work at all", "Totally useless", "This is frustrating", "I will never buy this again",
    "Horrible quality", "The service was bad", "This is not what I expected", "Waste of money", 
    "Very dissatisfied", "I regret buying this", "I feel scammed", "This is garbage"

     "It's okay", "Average service", "Not good, not bad", "Nothing special", 
    "It's just fine", "Could be better", "Meh, I guess", "The product is okay", 
    "Neither good nor bad", "It's just alright", "Okay but not great", "Not too bad", 
    "Could have been better", "It works, but I expected more", "I'm indifferent", 
    "Not impressed, not disappointed", "It does the job", "Mediocre at best", "Nothing outstanding", 
    "It’s not great, but it’s not terrible either"
]

labels = [
    'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 
    'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos',
    
    'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 
    'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg'

    'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 
    'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu', 'neu'
]

# Ensure that texts and labels have the same length
assert len(texts) == len(labels), f"Number of texts: {len(texts)}, Number of labels: {len(labels)}"

# Vectorizer and model training
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# ------------------ Prediction Function ------------------
def predict_sentiment(text):
    x = vectorizer.transform([text])
    result = model.predict(x)[0]
    return result

# ------------------ GUI Setup ------------------
def analyze_sentiment():
    result_label.config(text="")
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return
    sentiment = predict_sentiment(user_input)
    
    if sentiment == "pos":
        emoji = "😊"
        color = "green"
        label = "Positive"
    elif sentiment == "neg":
        emoji = "😠"
        color = "red"
        label = "Negative"
    else:
        emoji = "😐"
        color = "orange"
        label = "Neutral"
    
    result_label.config(text=f"{emoji}  {label}", fg=color)

# ------------------ Tkinter Window ------------------
window = tk.Tk()
window.title("Sentiment Analyzer")
window.geometry("400x250")
window.configure(bg="#f0f0f0")

tk.Label(window, text="Enter a sentence to analyze sentiment:", font=("Arial", 12), bg="#f0f0f0").pack(pady=10)
entry = tk.Entry(window, width=50, font=("Arial", 11))
entry.pack(pady=5)

tk.Button(window, text="Analyze Sentiment", command=analyze_sentiment,
          bg="#4CAF50", fg="white", font=("Arial", 11), width=20).pack(pady=15)

result_label = tk.Label(window, text="", font=("Arial", 16, "bold"), bg="#f0f0f0")
result_label.pack(pady=10)

window.mainloop()

