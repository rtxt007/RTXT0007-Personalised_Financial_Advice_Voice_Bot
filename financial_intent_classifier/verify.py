import numpy as np
from tensorflow.keras.models import load_model
import joblib

class FinancialIntentClassifier:
    def __init__(self, model_dir):
        self.model = load_model(f"{model_dir}/model.h5")
        self.tfidf = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
        self.le = joblib.load(f"{model_dir}/label_encoder.pkl")
        with open(f"{model_dir}/classes.txt") as f:
            self.classes = f.read().splitlines()
    
    def predict(self, text):
        # Preprocess input
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()
        vectorized = self.tfidf.transform([cleaned]).toarray()
        # Predict
        probabilities = self.model.predict(vectorized, verbose=0)
        return self.classes[np.argmax(probabilities)]

# Example usage
if __name__ == "__main__":
    classifier = FinancialIntentClassifier(".")
    sample_text = "What's the current price of Bitcoin?"
    print(f"Prediction: {classifier.predict(sample_text)}")
