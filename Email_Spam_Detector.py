from joblib import load
import re

# Load the trained KNN model
model_path = "KNN_best_model.pkl"
model = load(model_path)




def extract_features_from_email(raw_text):
    """
    Extract features from raw email text based on the Spambase dataset's attributes.
    """
    # Define the words and characters for which we'll compute frequencies
    words = [
        "make", "address", "all", "3d", "our", "over", "remove", "internet",
        "order", "mail", "receive", "will", "people", "report", "addresses",
        "free", "business", "email", "you", "credit", "your", "font", "000",
        "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
        "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
        "cs", "meeting", "original", "project", "re", "edu", "table", "conference"
    ]

    chars = [";", "(", "[", "!", "$", "#"]

    # Preprocess the raw text
    text = raw_text.lower()
    num_words = len(re.findall(r'\w+', text))
    num_chars = len(text)

    # Extract word and char frequencies
    word_freqs = [100 * text.count(word) / num_words if num_words > 0 else 0 for word in words]
    char_freqs = [100 * text.count(char) / num_chars if num_chars > 0 else 0 for char in chars]

    # Extract capital letter run attributes
    capital_runs = re.findall(r'[A-Z]+', raw_text)
    capital_run_lengths = [len(run) for run in capital_runs]
    avg_capital_run_length = sum(capital_run_lengths) / len(capital_run_lengths) if capital_run_lengths else 0
    max_capital_run_length = max(capital_run_lengths) if capital_run_lengths else 0
    total_capital_letters = sum(capital_run_lengths)

    # Compile the feature vector
    features = word_freqs + char_freqs + [avg_capital_run_length, max_capital_run_length, total_capital_letters]

    return features


# Test the function with a sample email text
sample_email = input("Please paste email text here to detect spam :")
extracted_features = extract_features_from_email(sample_email)
extracted_features[:10]  # Display the first 10 features for review


def predict_spam_or_ham(raw_email_text):
    """
    Predicts whether a raw email text is spam or ham using the trained KNN model.
    """
    # Extract features from the raw email text
    features = extract_features_from_email(raw_email_text)

    # Use the model to make a prediction
    prediction = model.predict([features])[0]

    return "SPAM" if prediction == 1 else "HAM"


# Test the prediction function with the sample email text
prediction_result = predict_spam_or_ham(sample_email)
print(f"\n\n\nPrediction Results : {prediction_result}")
