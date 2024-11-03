import torch
from transformers import Trainer, TrainingArguments

class TrainerModule:
    def __init__(self, model, train_data, train_labels, val_data, val_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=(self.train_data, self.train_labels),
            eval_dataset=(self.val_data, self.val_labels)
        )

        trainer.train()
