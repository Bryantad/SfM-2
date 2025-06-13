"""
Phase 4: SFM-2 Evaluation Script
Evaluates SFM-2 model on BLEU, syntax accuracy, and function completion rate.
- Loads model and tokenizer from models/sfm-2/
- Evaluates on datasets/cleaned/*.sona
- Saves results to eval/sfm2_eval_results.json
"""
import os
import json
import argparse
from glob import glob
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/sfm-2/"))
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/cleaned/"))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "sfm2_eval_results.json"))


def evaluate(model_dir, tokenizer_path, data_dir, results_path):
    """Run the evaluation loop."""
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model.eval()

    bleu_scores = []
    syntax_correct = 0
    func_complete = 0
    n = 0

    for file in glob(os.path.join(data_dir, "*.sona")):
        with open(file, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        if len(ref) < 20:
            continue
        inputs = tokenizer(ref, return_tensors="pt")
        output_ids = model.generate(
            inputs["input_ids"], max_length=inputs["input_ids"].shape[1] + 64, do_sample=False
        )
        gen = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu)
        if gen.count("{") == gen.count("}") and gen.count("(") == gen.count(")"):
            syntax_correct += 1
        if "fn" in gen and gen.strip().endswith("}"):
            func_complete += 1
        n += 1

    results = {
        "bleu_mean": sum(bleu_scores) / max(1, len(bleu_scores)),
        "syntax_accuracy": syntax_correct / max(1, n),
        "function_completion_rate": func_complete / max(1, n),
        "samples": n,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ SFM-2 evaluation complete. Results saved to {results_path}")


def main(argv=None):
    """Entry point for the ``sfm2-evaluate`` console script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SFM-2 model")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory containing the trained model")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, help="Path to the tokenizer file")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory of evaluation data")
    parser.add_argument("--output", default=RESULTS_PATH, help="File to write evaluation metrics")
    args = parser.parse_args(argv)

    evaluate(args.model_dir, args.tokenizer, args.data_dir, args.output)


if __name__ == "__main__":
    main()
