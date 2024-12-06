from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import T5Tokenizer

def preprocess_dataset(dataset_name, split_name, tokenizer, max_length=128):
    """
    Preprocess a dataset by loading, extracting, and tokenizing complex and simple sentences.
    Args:
        dataset_name (str): Name of the dataset to load.
        split_name (str): Split to load (e.g., "part_1").
        tokenizer: Tokenizer for tokenizing text.
        max_length (int): Maximum token length for inputs and labels.
    Returns:
        Dataset: Tokenized dataset ready for training.
    """
    dataset = load_dataset(dataset_name, split=split_name)

    def tokenize_function(example):
        try:
            # Flatten sentences
            complex_sentences = example["normal"]["normal_article_content"]["normal_sentence"]
            simple_sentences = example["simple"]["simple_article_content"]["simple_sentence"]

            complex_text = " ".join(complex_sentences)
            simple_text = " ".join(simple_sentences)

            # Tokenize
            inputs = tokenizer("simplify for beginners: " + complex_text, max_length=128, truncation=True, padding="max_length")
            labels = tokenizer(simple_text, max_length=128, truncation=True, padding="max_length")

            inputs["labels"] = labels["input_ids"]
            return inputs
        except Exception as e:
            print(f"Error processing example: {e}")
            return {}  # Skip problematic examples

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    return tokenized_dataset

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Preprocess both parts of the dataset
    print("Processing Part 1...")
    tokenized_data_part1 = preprocess_dataset("wiki_auto", "part_1", tokenizer)

    print("Processing Part 2...")
    tokenized_data_part2 = preprocess_dataset("wiki_auto", "part_2", tokenizer)

    # Combine datasets
    print("Combining datasets...")
    combined_data = concatenate_datasets([tokenized_data_part1, tokenized_data_part2])

    # Create train-validation split
    print("Splitting dataset...")
    split_data = combined_data.train_test_split(test_size=0.1, shuffle=True, seed=42)

    # Save the full dataset
    print("Saving full dataset...")
    dataset_dict = DatasetDict({
        "train": split_data["train"],
        "validation": split_data["test"]
    })
    dataset_dict.save_to_disk("../data/processed/wiki_auto")

    # Create a smaller dataset for testing/training
    print("Creating small dataset...")
    small_train_data = split_data["train"].select(range(2000))  # First 2000 examples
    small_val_data = split_data["test"].select(range(500))      # First 500 examples

    # Save the smaller dataset
    print("Saving small dataset...")
    small_dataset_dict = DatasetDict({
        "train": small_train_data,
        "validation": small_val_data
    })
    small_dataset_dict.save_to_disk("../data/processed/small_wiki_auto")
