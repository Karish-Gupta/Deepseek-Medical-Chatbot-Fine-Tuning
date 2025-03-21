from transformers import TrainingArguments, Trainer

class Model:
   def __init__(self, model, tokenizer, train_dataset, test_dataset, lr, num_epochs, weight_decay, batch_size):
      self.model = model
      self.tokenizer = tokenizer
      self.train_dataset = train_dataset
      self.test_dataset = test_dataset
      self.lr = lr
      self.num_epochs = num_epochs
      self.weight_decay = weight_decay
      self.batch_size = batch_size
   
   def train(self):
      training_args = TrainingArguments(
      output_dir="./results",         # Save model checkpoints
      evaluation_strategy="epoch",    # Evaluate after each epoch
      save_strategy="epoch",          # Save model after each epoch
      per_device_train_batch_size=self.batch_size,  # Adjust based on available GPU memory
      per_device_eval_batch_size=self.batch_size,
      gradient_accumulation_steps=4,  # Helps with low GPU memory
      learning_rate=self.lr,
      weight_decay=self.weight_decay,
      num_train_epochs=self.num_epochs,             # Adjust as needed
      logging_dir="./logs",           # Log directory
      logging_steps=100,
      save_total_limit=2,             # Keep only last 2 checkpoints
      fp16=True,                      # Use mixed precision for efficiency
      report_to="none",              # Disable logging to external services
      )

      trainer = Trainer(
         model=self.model,
         args=training_args,
         train_dataset=self.train_dataset,
         eval_dataset=self.test_dataset,
         tokenizer=self.tokenizer
      )

      trainer.train()
