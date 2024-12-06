import random
from datasets import Dataset

def introduce_typos(text, num_typos=2):
    """
    Introduce random typos into the given text.
    """
    words = text.split()
    for _ in range(num_typos):
        if not words:
            break
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        if len(word) > 1:
            char_idx = random.randint(0, len(word) - 1)
            new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
            word = word[:char_idx] + new_char + word[char_idx + 1:]
            words[word_idx] = word
    return " ".join(words)

def replace_with_synonyms(text, synonym_dict):
    """
    Replace words in the text with their synonyms using a predefined dictionary.
    """
    words = text.split()
    replaced_words = [synonym_dict.get(word, word) for word in words]
    return " ".join(replaced_words)

def add_noise_to_dataset_batched(dataset, perturbation_func, tokenizer, batch_size=64, **kwargs):
    """
    Add noise to a dataset in batches.
    """
    perturbed_examples = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Convert batch to a list of examples
        batch_examples = [example for example in batch]
        
        # Check if 'input_ids' or 'text' exists in the first example
        if "input_ids" in batch_examples[0]:
            input_texts = [tokenizer.decode(example["input_ids"], skip_special_tokens=True) for example in batch_examples]
            labels = [example["labels"] for example in batch_examples]
        elif "text" in batch_examples[0]:
            input_texts = [example["text"] for example in batch_examples]
            labels = [example.get("labels", None) for example in batch_examples]
        else:
            raise KeyError("Dataset does not contain 'input_ids' or 'text'. Check the dataset format.")
        
        # Apply perturbation
        perturbed_texts = [perturbation_func(text, **kwargs) for text in input_texts]
        
        # Tokenize perturbed texts
        encoded_inputs = tokenizer(perturbed_texts, truncation=True, padding="max_length", max_length=128)
        
        # Build perturbed examples
        for idx in range(len(perturbed_texts)):
            perturbed_example = {
                "input_ids": encoded_inputs["input_ids"][idx],
                "attention_mask": encoded_inputs["attention_mask"][idx],
                "labels": labels[idx],
            }
            perturbed_examples.append(perturbed_example)
    
    # Create new Dataset
    perturbed_dataset = Dataset.from_list(perturbed_examples)
    return perturbed_dataset
