from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk

# Load the trained model and tokenizer
model_path = "../data/checkpoints/wiki_auto"  # Update with your checkpoint path if different
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to simplify text
def simplify_text(input_text, model, tokenizer, max_length=128, num_beams=5):
    """
    Simplify the given text using the trained model.
    Args:
        input_text (str): The text to simplify.
        model: The trained T5 model.
        tokenizer: The tokenizer for the model.
        max_length (int): Maximum length of the generated text.
        num_beams (int): Number of beams for beam search.
    Returns:
        str: Simplified text.
    """
    inputs = tokenizer("simplify for beginners: " + input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load a small dataset for testing
test_data_path = "../data/processed/small_wiki_auto"  # Update with your test dataset path
dataset = load_from_disk(test_data_path)["validation"]  # Use the validation split for testing

# Test the model on a few examples
print("\nTesting the model on validation examples:")
for i, example in enumerate(dataset.select(range(5))):  # Testing on the first 5 examples
    input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    reference_text = tokenizer.decode(example["labels"], skip_special_tokens=True)

    simplified = simplify_text(input_text, model, tokenizer)
    print(f"Example {i + 1}:")
    print(f"Input: {input_text}")
    print(f"Reference Simplification: {reference_text}")
    print(f"Model Simplification: {simplified}\n")
