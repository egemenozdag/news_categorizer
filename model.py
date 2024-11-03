import torch
from transformers import AutoModelForSequenceClassification


class NewsCategorizerModel:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)

    def get_model(self):
        return self.model
