from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk, Dataset
from perturbation_utils import introduce_typos, replace_with_synonyms, add_noise_to_dataset_batched
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_kincaid_grade
import torch

def evaluate_model_on_dataset(model, tokenizer, dataset, batch_size=64):
    """
    Evaluate the model on a given dataset with BLEU and Flesch-Kincaid metrics.
    """
    smoothing_function = SmoothingFunction().method1
    total_bleu = 0
    total_fk = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch_examples = [example for example in batch]

            input_ids = torch.tensor([example["input_ids"] for example in batch_examples]).cuda()
            attention_mask = torch.tensor([example["attention_mask"] for example in batch_examples]).cuda()
            labels = [example["labels"] for example in batch_examples]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            reference_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            for ref, gen in zip(reference_texts, generated_texts):
                total_bleu += sentence_bleu(
                    [ref.split()], gen.split(), smoothing_function=smoothing_function
                )
                total_fk += flesch_kincaid_grade(gen)
                total_samples += 1

    average_bleu = total_bleu / total_samples if total_samples > 0 else 0
    average_fk = total_fk / total_samples if total_samples > 0 else 0

    return average_bleu, average_fk

if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model_path = "../data/checkpoints/wiki_auto"
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    print("Loading dataset...")
    dataset = load_from_disk("../data/processed/wiki_auto")["validation"]

    # Ensure the dataset is in the correct format
    processed_dataset = dataset.map(lambda x: x)

    perturbations = [
        (introduce_typos, {"num_typos": 2}),
        (replace_with_synonyms, {"synonym_dict": {"difficult": "hard", "simple": "easy"}}),
    ]

    for perturbation_func, kwargs in perturbations:
        print(f"\nApplying perturbation: {perturbation_func.__name__}")
        perturbed_dataset = add_noise_to_dataset_batched(
            processed_dataset, perturbation_func, tokenizer, batch_size=64, **kwargs
        )
        bleu, fk = evaluate_model_on_dataset(model, tokenizer, perturbed_dataset)
        print(f"Perturbation: {perturbation_func.__name__}, BLEU: {bleu:.4f}, FK: {fk:.4f}")
