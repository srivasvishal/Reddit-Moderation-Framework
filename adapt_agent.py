"""
Adapt Agent with LoRA Adapters

Fine-tunes BERT model for toxicity detection using PEFT LoRA adapters.
Trains subreddit-specific adapters from human feedback.

References I Used:
- LoRA Blog: https://kalomaze.bearblog.dev/rl-lora-ddd/
- PEFT: https://huggingface.co/docs/peft/index
- BERT: https://huggingface.co/google-bert/bert-base-uncased
"""

import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from text_processing import tokenize_for_bert, TOXICITY_THRESHOLD


class AdaptAgent:
    def __init__(
        self,
        model_name="bert-base-uncased",
        lora_rank=8,
        lora_alpha=64,
        lora_dropout=0.1,
        adapter_save_path=None,
        device=None,
        modules_to_save=None
    ):
        agents_dir = Path(__file__).parent

        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Adapter save path: models/lora_adapters/
        if adapter_save_path is None:
            self.adapter_save_path = agents_dir / "models" / "lora_adapters"
        elif Path(adapter_save_path).is_absolute():
            self.adapter_save_path = Path(adapter_save_path)
        else:
            self.adapter_save_path = agents_dir / adapter_save_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.adapter_save_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.base_model = BertForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value", "dense",],# "classifier"],  # Standard BERT pattern
            modules_to_save=modules_to_save,  # Empty = pure LoRA (blog default)
            bias="none",
            inference_mode=False
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        print(f"AdaptAgent initialized on {self.device} (rank={lora_rank}, alpha={lora_alpha})")

    def _prepare_training_data(self, suggestions):
        """
        Prepare training data with BERT tokenization.
        """
        input_ids = []
        attention_masks = []
        labels = []

        for suggestion in suggestions:
            # Tokenize with preprocessing (uses shared utility)
            encoded = tokenize_for_bert(
                suggestion["text"],
                self.tokenizer,
                preprocess=True
            )

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

            # Auto-threshold continuous scores to binary
            toxicity_score = suggestion["toxicity_score"]
            binary_label = 1 if toxicity_score >= TOXICITY_THRESHOLD else 0
            labels.append(binary_label)

        # Convert to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return TensorDataset(input_ids, attention_masks, labels)

    def adapt_feedback_only(
        self,
        feedback_samples,
        epochs=3,
        batch_size=8,
        learning_rate=2e-4,
        validation_split=None,
        save_after_training=True,
        adapter_name=None,
        tensorboard_run_name=None
    ):
        """
        Train LoRA adapter using ONLY feedback samples (no base CSV mixing).
        """
        if len(feedback_samples) < 2:
            raise ValueError(f"Need at least 2 feedback samples, got {len(feedback_samples)}")

        # Shuffle samples (random already imported at module level)
        shuffled = feedback_samples.copy()
        random.shuffle(shuffled)

        # Split into train/validation if requested
        if validation_split and validation_split > 0:
            split_idx = int(len(shuffled) * (1 - validation_split))
            train_samples = shuffled[:split_idx]
            val_samples = shuffled[split_idx:]
            print(f"\n{'='*60}")
            print("Feedback-Only LoRA Training (with validation)")
            print(f"{'='*60}")
            print(f"Total samples: {len(feedback_samples)}")
            print(f"Train samples: {len(train_samples)} ({(1-validation_split)*100:.0f}%)")
            print(f"Val samples: {len(val_samples)} ({validation_split*100:.0f}%)")
        else:
            train_samples = shuffled
            val_samples = None
            print(f"\n{'='*60}")
            print("Feedback-Only LoRA Training (no validation split)")
            print(f"{'='*60}")
            print(f"Samples: {len(feedback_samples)} (all used for training)")

        print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")

        # Prepare datasets
        train_dataset = self._prepare_training_data(train_samples)
        val_dataset = self._prepare_training_data(val_samples) if val_samples else None

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=min(batch_size, len(train_dataset))
        )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=min(batch_size, len(val_dataset))
            )

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            eps=1e-8
        )

        # Setup scheduler
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Setup TensorBoard writer
        writer = None
        if tensorboard_run_name:
            log_dir = Path('runs') / tensorboard_run_name
            writer = SummaryWriter(str(log_dir))
            print(f"TensorBoard logging to: {log_dir}")

        # Training loop
        training_stats = []
        best_val_accuracy = 0
        global_step = 0

        for epoch_i in range(epochs):
            print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")

            # Training phase
            self.model.train()
            total_train_loss = 0

            for batch in train_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                optimizer.zero_grad()
                output = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

                loss = output.loss
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Log batch loss to TensorBoard
                if writer:
                    writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                global_step += 1

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"  Train Loss: {avg_train_loss:.4f}")

            # Log epoch train loss to TensorBoard
            if writer:
                writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch_i)

            epoch_stats = {
                'epoch': epoch_i + 1,
                'train_loss': avg_train_loss
            }

            # Validation phase (if validation set exists)
            if val_dataloader:
                self.model.eval()
                total_val_loss = 0
                total_val_correct = 0
                total_val_samples = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        b_input_ids = batch[0].to(self.device)
                        b_input_mask = batch[1].to(self.device)
                        b_labels = batch[2].to(self.device)

                        output = self.model(
                            b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels
                        )

                        total_val_loss += output.loss.item()

                        # Calculate accuracy
                        preds = torch.argmax(output.logits, dim=1)
                        total_val_correct += (preds == b_labels).sum().item()
                        total_val_samples += b_labels.size(0)

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_accuracy = total_val_correct / total_val_samples

                print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                # Log validation metrics to TensorBoard
                if writer:
                    writer.add_scalar('Loss/validation', avg_val_loss, epoch_i)
                    writer.add_scalar('Accuracy/validation', val_accuracy, epoch_i)

                epoch_stats['val_loss'] = avg_val_loss
                epoch_stats['val_accuracy'] = val_accuracy

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy

            training_stats.append(epoch_stats)

        # Close TensorBoard writer
        if writer:
            writer.close()
            print(f"TensorBoard logs saved")

        # Save final model after all epochs
        if save_after_training and adapter_name:
            save_path = self.save_adapter(adapter_name)
            print(f"\nAdapter saved: {save_path}")

        final_loss = training_stats[-1]['train_loss']
        print(f"\nTraining complete! Final loss: {final_loss:.4f}")

        result = {
            "training_stats": training_stats,
            "final_loss": final_loss,
            "samples_trained": len(train_dataset),
            "epochs": epochs
        }

        if val_dataloader:
            result["best_val_accuracy"] = best_val_accuracy
            result["final_val_accuracy"] = training_stats[-1].get('val_accuracy', 0)

        return result

    def predict(self, comment, T=1.0):
        """
        Predict toxicity for a single comment.
        """
        original_comment = comment

        # Tokenize with preprocessing (uses shared utility)
        encoded = tokenize_for_bert(comment, self.tokenizer, preprocess=True)

        self.model.eval()

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids,
                token_type_ids=None,
                attention_mask=attention_mask
            )

        logits = output.logits.detach().cpu().numpy()
        prediction = np.argmax(logits, axis=1).flatten()

        # Temperature-scaled confidence (softmax)
        exp_logits = np.exp(logits / T)
        confidence = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        confidence = confidence[0][prediction]

        return logits, int(prediction[0]), float(confidence[0]), original_comment

    def save_adapter(self, version_name):
        if version_name is None:
            version_name = f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        save_path = self.adapter_save_path / version_name
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def load_finetuned_weights(self, weights_path=None):
        """Load finetuned BERT weights. Defaults to bert_model_full_data."""
        if weights_path is None:
            weights_path = Path(__file__).parent / "models" / "bert_model_full_data"
        else:
            weights_path = Path(weights_path)

        print(f"Loading finetuned model from: {weights_path}")
        self.base_model = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.base_model.to(self.device)

        print("Finetuned model loaded successfully")
