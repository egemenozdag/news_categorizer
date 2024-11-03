import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class DataLoader:
    def __init__(self, api_key, query='news', model_path='---'):
        self.url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.train_data = None
        self.val_data = None
        self.data = None

    def scrape_data(self):
        with open('new_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        articles = data.get('articles', [])
        if not isinstance(articles, list):
            raise ValueError("JSON yapısı beklenen formatta değil.")

        processed_articles = []
        for item in articles:
            title = item.get('title')
            content = item.get('description')
            if title and content:
                label = self.label_article(title)
                processed_articles.append({'text': content, 'label': label})

        self.data = pd.DataFrame(processed_articles)

        if not self.data.empty:
            self.train_data, self.val_data = train_test_split(self.data, test_size=0.2, random_state=42)
            print(f"Training set: {len(self.train_data)}, Validation set: {len(self.val_data)}")
        else:
            raise ValueError("Veri çerçevesi boş. Veriler doğru yüklenmedi.")

    def label_article(self, title):
        if "sports" in title.lower(): return 0
        elif "economy" in title.lower(): return 1
        elif "politics" in title.lower(): return 2
        else: return -1

    def tokenize_data(self, data):
        return self.tokenizer(
            data['text'].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

    def get_train_val_data(self):
        if self.train_data is None or self.val_data is None:
            raise ValueError("Eğitim ve doğrulama verileri bulunamadı. scrape_data fonksiyonunu çalıştırmayı deneyin.")

        return self.tokenize_data(self.train_data), self.train_data['label'].tolist(), \
               self.tokenize_data(self.val_data), self.val_data['label'].tolist()
