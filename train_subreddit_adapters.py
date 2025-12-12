"""
Subreddit-Specific LoRA Adapter Training. Scan through the human feedback CSV and train a separate LoRA adapter for each subreddit that meets the minimum sample size.
"""

import argparse
import pandas as pd
from pathlib import Path
from adapt_agent import AdaptAgent


def load_feedback_by_subreddit(csv_path):
    """
    Load human feedback CSV and group by subreddit.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Human feedback CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Verify required columns
    required = ['comment_text', 'human_label', 'subreddit']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Group by subreddit
    subreddit_feedback = {}

    for _, row in df.iterrows():
        subreddit = row['subreddit']
        text = row['comment_text']
        label = row['human_label']

        # Skip invalid entries
        if pd.isna(subreddit) or pd.isna(text) or pd.isna(label):
            continue
        if len(str(text).strip()) < 3:
            continue

        subreddit = str(subreddit).strip()
        if not subreddit:
            subreddit = "unknown"

        # Convert label to toxicity score
        label_lower = str(label).lower().strip()
        if label_lower == 'toxic':
            toxicity_score = 1.0
        elif label_lower == 'non-toxic':
            toxicity_score = 0.0
        else:
            continue  # Skip unknown labels

        if subreddit not in subreddit_feedback:
            subreddit_feedback[subreddit] = []

        subreddit_feedback[subreddit].append({
            "text": str(text),
            "toxicity_score": toxicity_score
        })

    return subreddit_feedback


def print_subreddit_summary(subreddit_feedback):
    """Print summary of feedback per subreddit."""
    print("\n" + "=" * 60)
    print("SUBREDDIT FEEDBACK SUMMARY")
    print("=" * 60)

    total_samples = 0
    for subreddit, samples in sorted(subreddit_feedback.items()):
        toxic_count = sum(1 for s in samples if s['toxicity_score'] == 1.0)
        safe_count = len(samples) - toxic_count
        total_samples += len(samples)
        print(f"  r/{subreddit:30} | {len(samples):3} samples (toxic: {toxic_count}, safe: {safe_count})")

    print("-" * 60)
    print(f"  {'TOTAL':30} | {total_samples:3} samples")
    print("=" * 60 + "\n")

    return total_samples


def preprocess_with_triage(subreddit_feedback):
    """
    Preprocess feedback samples with triage agent.
    """
    try:
        from triage_agent import TriageAIAgent
        triage = TriageAIAgent()

        if not triage.ai_expander.api_available:
            print("WARNING: Ollama not available, skipping triage preprocessing")
            return subreddit_feedback

        print("\nPreprocessing training data with triage agent...")
        total_samples = sum(len(samples) for samples in subreddit_feedback.values())
        processed = 0
        total_fixes = 0

        for subreddit, samples in subreddit_feedback.items():
            for sample in samples:
                processed += 1
                if processed % 50 == 0 or processed == total_samples:
                    print(f"  Triage progress: {processed}/{total_samples} samples...")

                record = {'comment_body': sample['text']}
                cleaned_record = triage.clean_record(record)
                sample['text'] = cleaned_record.get('comment_body', sample['text'])
                fixes = cleaned_record.get('_fixes_applied', 0)
                total_fixes += fixes

        print(f"Triage preprocessing complete: {total_fixes} fixes applied to {total_samples} samples")
        return subreddit_feedback

    except Exception as e:
        print(f"WARNING: Triage preprocessing failed: {e}")
        return subreddit_feedback


def train_subreddit_adapter(
    agent,
    subreddit,
    feedback_samples,
    epochs,
    batch_size,
    learning_rate,
    validation_split=0.2
):
    """
    Train a LoRA adapter for a specific subreddit using feedback-only training.

    LoRA preserves base model weights (frozen), so we only train on human feedback.
    The base model's general toxicity knowledge is inherently preserved.
    """
    adapter_name = f"reddit_{subreddit.lower()}_adapter"

    print(f"\n{'='*60}")
    print(f"Training adapter for r/{subreddit}")
    print(f"{'='*60}")
    print(f"  Feedback samples: {len(feedback_samples)}")
    print(f"  Validation split: {validation_split}")
    print(f"  Adapter name: {adapter_name}")

    try:
        # Use feedback-only training with validation split
        tensorboard_run_name = f"reddit_{subreddit.lower()}_adapter"
        result = agent.adapt_feedback_only(
            feedback_samples=feedback_samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            save_after_training=True,
            adapter_name=adapter_name,
            tensorboard_run_name=tensorboard_run_name
        )

        result['adapter_name'] = adapter_name
        result['subreddit'] = subreddit

        return result

    except Exception as e:
        print(f"\n  ERROR training r/{subreddit}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train subreddit-specific LoRA adapters from human feedback"
    )
    parser.add_argument(
        "--min-samples", type=int, default=300,
        help="Minimum feedback samples required to train an adapter (default: 300)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs per adapter (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--validation-split", type=float, default=0.2,
        help="Validation split ratio (default: 0.2 = 80/20 train/val)"
    )
    parser.add_argument(
        "--use-triage", action="store_true", default=False,
        help="Enable triage preprocessing (requires Ollama running)"
    )
    parser.add_argument(
        "--feedback-csv", type=str, default="None",
        help="Path to human_feedback.csv (default: dashboard_data/human_feedback.csv)"
    )

    args = parser.parse_args()

    # Resolve feedback CSV path
    if args.feedback_csv:
        csv_path = Path(args.feedback_csv)
    else:
        csv_path = Path(__file__).parent / "dashboard_data" / "human_feedback.csv"

    print("\n" + "=" * 60)
    print("SUBREDDIT-SPECIFIC LORA ADAPTER TRAINING")
    print("=" * 60)
    print(f"Mode: Feedback-only (LoRA preserves base weights)")
    print(f"Feedback CSV: {csv_path}")
    print(f"Min samples per subreddit: {args.min_samples}")
    print(f"Validation split: {args.validation_split} ({(1-args.validation_split)*100:.0f}% train / {args.validation_split*100:.0f}% val)")
    print(f"Triage preprocessing: {'Enabled' if args.use_triage else 'Disabled'}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    # Load and group feedback by subreddit
    print("\nLoading human feedback...")
    subreddit_feedback = load_feedback_by_subreddit(csv_path)

    if not subreddit_feedback:
        print("ERROR: No valid feedback found in CSV!")
        return

    # Apply triage preprocessing if enabled
    if args.use_triage:
        subreddit_feedback = preprocess_with_triage(subreddit_feedback)

    # Print summary
    total_samples = print_subreddit_summary(subreddit_feedback)

    # Filter subreddits with enough samples
    eligible_subreddits = {
        sub: samples for sub, samples in subreddit_feedback.items()
        if len(samples) >= args.min_samples
    }

    skipped_subreddits = {
        sub: samples for sub, samples in subreddit_feedback.items()
        if len(samples) < args.min_samples
    }

    if skipped_subreddits:
        print(f"Skipping {len(skipped_subreddits)} subreddit(s) with < {args.min_samples} samples:")
        for sub, samples in skipped_subreddits.items():
            print(f"- r/{sub} ({len(samples)} samples)")

    if not eligible_subreddits:
        print(f"\nERROR: No subreddits have >= {args.min_samples} samples!")
        print("Try lowering --min-samples or collect more feedback.")
        return

    print(f"\nWill train {len(eligible_subreddits)} adapter(s):")
    for sub in eligible_subreddits:
        print(f"- r/{sub}")

    # Check once if finetuned weights exist
    finetuned_weights_path = Path(__file__).parent / "models" / "bert_model_full_data"
    use_finetuned = finetuned_weights_path.exists()

    print("\n" + "-" * 60)
    print("AdaptAgent Configuration")
    print("-" * 60)
    if use_finetuned:
        print(f"Will use finetuned weights from: {finetuned_weights_path}")
    else:
        print("Will use base BERT weights (finetuned model not found)")

    # Train adapter for each eligible subreddit
    results = []

    for subreddit, samples in eligible_subreddits.items():
        # Create fresh agent for each subreddit
        # This ensures LoRA adapters don't accumulate across trainings
        agent = AdaptAgent()

        if use_finetuned:
            try:
                agent.load_finetuned_weights()
            except FileNotFoundError:
                print(f"  Warning: Failed to load finetuned weights for r/{subreddit}")

        result = train_subreddit_adapter(
            agent=agent,
            subreddit=subreddit,
            feedback_samples=samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split
        )

        if result:
            results.append(result)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Subreddits processed: {len(eligible_subreddits)}")
    print(f"Adapters trained successfully: {len(results)}")

    if results:
        print("\nTrained adapters:")
        for r in results:
            val_acc = r.get('best_val_accuracy')
            if val_acc is not None:
                print(f"  r/{r['subreddit']:20} | loss: {r['final_loss']:.4f} | val_acc: {val_acc:.4f} | samples: {r['samples_trained']}")
            else:
                print(f"  r/{r['subreddit']:20} | loss: {r['final_loss']:.4f} | samples: {r['samples_trained']}")

    if len(results) < len(eligible_subreddits):
        failed = len(eligible_subreddits) - len(results)
        print(f"\nWARNING: {failed} adapter(s) failed to train")

    print("\n" + "=" * 60)
    print(f"Adapters saved to: {agent.adapter_save_path}")
    print(f"TensorBoard logs: runs/reddit_<subreddit>_adapter/")
    print("Run: tensorboard --logdir=runs")
    print("=" * 60 + "\n")


"""
Usage:
    python train_subreddit_adapters.py
    python train_subreddit_adapters.py --min-samples 5
    python train_subreddit_adapters.py --epochs 2 --batch-size 16
"""
if __name__ == "__main__":
    main()
