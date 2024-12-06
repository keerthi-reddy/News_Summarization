import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch

# Load the preprocessed dataset
data_dir = "../data/processed/wiki_auto"
print("Loading dataset...")
dataset = load_from_disk(data_dir)

# Load the T5 model and tokenizer
model_name = "t5-small"
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the data collator to handle tokenization padding
def data_collator(batch):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    # Convert to PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    # Pad labels manually and convert to tensors
    max_length = max(len(label) for label in labels)
    padded_labels = [
        label + [-100] * (max_length - len(label)) for label in labels
    ]
    labels = torch.tensor(padded_labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Define training arguments
output_dir = "../data/checkpoints/wiki_auto"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=3e-5,  # Reduce learning rate for finer adjustments
    per_device_train_batch_size=16,  # Increase batch size for faster convergence
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Train for more epochs to improve results
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_strategy="epoch",
    save_total_limit=2,
    seed=42,
    fp16=True,  # Enable mixed precision for faster training on compatible GPUs
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()

    print("Saving the model...")
    trainer.save_model(output_dir)

    print("Training complete. Model saved to:", output_dir)
