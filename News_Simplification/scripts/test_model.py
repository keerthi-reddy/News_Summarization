from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("../data/checkpoints/wiki_auto_cpu")
tokenizer = T5Tokenizer.from_pretrained("../data/checkpoints/wiki_auto_cpu")

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
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    # Generate output
    outputs = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=5,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=2
    )
    
    # Decode and return the simplified text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with custom examples
examples = [
    "simplify: This is a complex sentence that requires simplification.",
    "simplify: The quick brown fox jumped over the lazy dog.",
    "simplify: The government introduced a new bill in parliament to improve education policies.",
]

for example in examples:
    simplified = simplify_text(example, model, tokenizer)
    print(f"Input: {example}")
    print(f"Simplified: {simplified}\n")

# Optionally test with training data for debugging
try:
    from datasets import load_from_disk

    # Load the small dataset used for training
    dataset = load_from_disk("../data/processed/small_wiki_auto")
    # Print a sample to check the structure
    print("Dataset structure:", dataset["train"][0])
    train_examples = dataset["train"][:5]  # Fetch first 5 training examples

    print("\nTesting with training data:")
    for i, example in enumerate(train_examples):
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        simplified = simplify_text(input_text, model, tokenizer)
        print(f"Training Example {i+1}:")
        print(f"Input: {input_text}")
        print(f"Simplified: {simplified}\n")
except Exception as e:
    print("Could not load training dataset for testing. Skipping this step.")
    print(f"Error: {e}")
