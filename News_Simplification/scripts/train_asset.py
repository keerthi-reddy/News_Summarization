from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model(pretrained_model_path, output_dir):
    """
    Fine-tune a pre-trained T5 model on the ASSET dataset.
    Args:
        pretrained_model_path (str): Path to pre-trained model checkpoint.
        output_dir (str): Directory to save fine-tuned model.
    """
    dataset = load_dataset("asset", split="train")
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    
    def tokenize_function(examples):
        inputs = ["simplify: " + text for text in examples["original"]]
        targets = examples["simplifications"][0]  # Use the first simplification
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_data = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        save_total_limit=2,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fine_tune_model("../data/checkpoints/wiki_auto", "../data/checkpoints/asset")