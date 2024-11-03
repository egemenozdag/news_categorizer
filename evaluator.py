from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, texts, labels):
        predictions = []
        for text in texts:
            pred = Predictor(self.model, self.tokenizer).predict(text)
            predictions.append(pred)
        print("Accuracy:", accuracy_score(labels, predictions))
        print("Classification Report:\n", classification_report(labels, predictions))
