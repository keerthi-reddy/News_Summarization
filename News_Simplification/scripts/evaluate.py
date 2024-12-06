from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import flesch_kincaid_grade  # For readability score
import torch


def custom_collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [example["attention_mask"] for example in batch]
    labels = [example["labels"] for example in batch]

    max_input_len = max(len(ids) for ids in input_ids)
    max_label_len = max(len(lbl) for lbl in labels)

    padded_input_ids = [ids + [0] * (max_input_len - len(ids)) for ids in input_ids]
    padded_attention_mask = [mask + [0] * (max_input_len - len(mask)) for mask in attention_mask]
    padded_labels = [lbl + [-100] * (max_label_len - len(lbl)) for lbl in labels]

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def evaluate_wikiauto(batch_size=64):
    print("Loading the fine-tuned WikiAuto model...")
    model = T5ForConditionalGeneration.from_pretrained("../data/checkpoints/wiki_auto").cuda()
    tokenizer = T5Tokenizer.from_pretrained("../data/checkpoints/wiki_auto")

    print("Loading WikiAuto validation dataset...")
    dataset = load_from_disk("../data/processed/wiki_auto")["validation"]

    smoothing_function = SmoothingFunction().method1
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    total_bleu = 0
    total_fk = 0  # For Flesch-Kincaid readability score
    total_samples = 0

    print("\nEvaluating the model on WikiAuto validation samples...")
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,  # Limit sentence length
                num_beams=3,  # Lower beam count for simpler outputs
                early_stopping=True,
                length_penalty=1.2,  # Penalize longer outputs
                no_repeat_ngram_size=3,  # Prevent redundancy
            )

            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            reference_texts = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]

            for ref, gen in zip(reference_texts, generated_texts):
                reference_tokens = [ref.split()]
                generated_tokens = gen.split()

                # Calculate BLEU score
                bleu_score = sentence_bleu(
                    reference_tokens,
                    generated_tokens,
                    smoothing_function=smoothing_function
                )
                total_bleu += bleu_score

                # Calculate Flesch-Kincaid readability score
                fk_score = flesch_kincaid_grade(gen)
                total_fk += fk_score

                total_samples += 1

    average_bleu = total_bleu / total_samples if total_samples > 0 else 0
    average_fk = total_fk / total_samples if total_samples > 0 else 0

    print(f"\nFinal Average BLEU Score on WikiAuto validation set: {average_bleu:.4f}")
    print(f"Final Average Flesch-Kincaid Grade Level on WikiAuto validation set: {average_fk:.4f}")


if __name__ == "__main__":
    evaluate_wikiauto(batch_size=64)
