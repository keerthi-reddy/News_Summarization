from datasets import load_dataset, concatenate_datasets, DatasetDict

from transformers import T5Tokenizer

def preprocess_dataset(dataset_name, split_name, tokenizer, max_length=128):
    dataset = load_dataset(dataset_name, split=split_name)

    def tokenize_function(examples):
        complex_sentences = examples["normal"]["normal_article_content"]["normal_sentence"]
        simple_sentences = examples["simple"]["simple_article_content"]["simple_sentence"]

        # Flatten sentences
        complex_text = " ".join(complex_sentences)
        simple_text = " ".join(simple_sentences)

        # Tokenize
        inputs = tokenizer("simplify: " + complex_text, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(simple_text, max_length=max_length, truncation=True, padding="max_length")

        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    return tokenized_dataset

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load and preprocess both parts
    tokenized_data_part1 = preprocess_dataset("wiki_auto", "part_1", tokenizer)
    tokenized_data_part2 = preprocess_dataset("wiki_auto", "part_2", tokenizer)

    # Combine datasets
    combined_data = concatenate_datasets([tokenized_data_part1, tokenized_data_part2])

    # Create train-validation split using Hugging Face's train_test_split
    split_data = combined_data.train_test_split(test_size=0.1, shuffle=True, seed=42)

    # Save full dataset
    dataset_dict = DatasetDict({
        "train": split_data["train"],
        "validation": split_data["test"]
    })
    dataset_dict.save_to_disk("../data/processed/wiki_auto")

    # Create a smaller dataset for faster CPU training
    small_train_data = split_data["train"].select(range(2000))  # First 2000 examples
    small_val_data = split_data["test"].select(range(500))      # First 500 examples

    # Save the smaller dataset
    small_dataset_dict = DatasetDict({
        "train": small_train_data,
        "validation": small_val_data
    })
    small_dataset_dict.save_to_disk("../data/processed/small_wiki_auto")
