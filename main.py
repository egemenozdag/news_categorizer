from data_loader import DataLoader
from model import NewsCategorizerModel
from trainer import TrainerModule
from predictor import Predictor
from evaluator import Evaluator

api_key = 'a03a10d2bae243b795a8a5ad5a8002fd'
data_loader = DataLoader(api_key=api_key)
data_loader.scrape_data()

train_data, train_labels, val_data, val_labels = data_loader.get_train_val_data()

model_class = NewsCategorizerModel()
model = model_class.get_model()

trainer = TrainerModule(model=model, train_data=train_data['input_ids'],
                        train_labels=train_labels, val_data=val_data['input_ids'],
                        val_labels=val_labels)
trainer.train()

predictor = Predictor(model=model_class.get_model(), tokenizer=data_loader.tokenizer)

sample_text = "Recent developments in the world of sports."
prediction = predictor.predict(sample_text)
print(f'Tahmin edilen kategori: {prediction}')

evaluator = Evaluator(model=model_class.get_model(), tokenizer=data_loader.tokenizer)
evaluator.evaluate(val_data['text'], val_labels)
