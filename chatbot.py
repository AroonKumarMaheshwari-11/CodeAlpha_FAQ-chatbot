import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # newer NLTK versions need this
from nltk.corpus import stopwords

faqs = [
    {
        "question": "hi",
        "answer": "Hi! I am FAQ-Chatbot AI. How can I help you?"
    },
    {
        "question": "can you tell about some artificial intelligence terms?",  # ✅ Fixed key
        "answer": "Yes! Sure, ask me freely."
    },
    {
        "question": "What is Artificial Intelligence?",
        "answer": "Artificial Intelligence (AI) is the simulation of human intelligence by machines to perform tasks like reasoning, learning, and problem-solving."
    },
    {
        "question": "What is Machine Learning?",
        "answer": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."
    },
    {
        "question": "What is Deep Learning?",
        "answer": "Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to analyze large amounts of data."
    },
    {
        "question": "What is a Neural Network?",
        "answer": "A Neural Network is a series of algorithms that mimic the human brain to recognize patterns and solve complex problems."
    },
    {
        "question": "What is Natural Language Processing?",
        "answer": "NLP (Natural Language Processing) is a branch of AI that helps computers understand, interpret, and generate human language."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is an AI-powered program designed to simulate conversation with users, typically used for customer service or FAQs."
    },
    {
        "question": "What is Python?",
        "answer": "Python is a high-level, easy-to-learn programming language widely used in AI, Machine Learning, and data science."
    },
    {
        "question": "What is data science?",
        "answer": "Data Science is a field that uses statistics, programming, and machine learning to extract insights from data."
    },
    {
        "question": "What is supervised learning?",
        "answer": "Supervised Learning is a type of ML where the model is trained on labeled data to predict outcomes for new data."
    },
    {
        "question": "What is unsupervised learning?",
        "answer": "Unsupervised Learning is a type of ML where the model finds hidden patterns in data without labeled responses."
    },
    {
        "question": "What is a large language model?",
        "answer": "A Large Language Model (LLM) is an AI model trained on massive text data to understand and generate human-like text. Examples: GPT, Claude."
    },
    {
        "question": "What is computer vision?",
        "answer": "Computer Vision is a field of AI that enables computers to interpret and understand visual information from images and videos."
    },
    {
        "question": "What is reinforcement learning?",
        "answer": "Reinforcement Learning is a type of ML where an agent learns by interacting with an environment and receiving rewards or penalties."
    },
    {
        "question": "What is overfitting?",
        "answer": "Overfitting occurs when a model learns the training data too well and performs poorly on new, unseen data."
    },
    {
        "question": "What is a dataset?",
        "answer": "A dataset is a collection of organized data used to train, test, and evaluate machine learning models."
    },
]

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)


# ✅ Vectorizer ek baar bahar banana — efficient
faq_questions = [preprocess(f["question"]) for f in faqs]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions)

def get_answer(user_input):
    cleaned_input = preprocess(user_input)
    user_vector = vectorizer.transform([cleaned_input])  # ✅ sirf transform
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I don't have an answer for that. Please try rephrasing your question."

    return faqs[best_match_index]["answer"]


print("=" * 50)
print("   FAQ Chatbot — Artificial Intelligence")
print("   Type 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue  # ✅ Empty input handle
    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a great day!")
        break
    answer = get_answer(user_input)
    print(f"Bot: {answer}")