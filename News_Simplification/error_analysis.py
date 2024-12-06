import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
import json

def error_analysis(dataset, model, tokenizer, num_examples=10):
    """
    Perform error analysis on a subset of the dataset.
    Saves errors in a JSON file for manual inspection.
    """
    errors = []
    model.eval()
    with torch.no_grad():
        for i, example in enumerate(dataset.select(range(num_examples))):
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            reference_text = tokenizer.decode(example["labels"][example["labels"] != -100], skip_special_tokens=True)

            inputs = tokenizer("simplify for beginners: " + input_text, return_tensors="pt", truncation=True).to("cuda")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if generated_text.strip() != reference_text.strip():
                errors.append({
                    "input": input_text,
                    "reference": reference_text,
                    "generated": generated_text,
                })

    # Save errors for manual inspection
    with open("error_analysis.json", "w") as f:
        json.dump(errors, f, indent=4)
    print("Error analysis saved to error_analysis.json")


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained("../data/checkpoints/wiki_auto").cuda()
    tokenizer = T5Tokenizer.from_pretrained("../data/checkpoints/wiki_auto")

    print("Loading dataset...")
    dataset = load_from_disk("../data/processed/wiki_auto")["validation"]

    print("Running error analysis...")
    error_analysis(dataset, model, tokenizer, num_examples=50)
