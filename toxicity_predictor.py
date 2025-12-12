import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import logging

from text_processing import tokenize_for_bert

logger = logging.getLogger(__name__)


class ToxicityPredictor:
    """
    Toxicity prediction with subreddit-specific LoRA adapter support.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.base_model = None
        self.current_adapter = None  # Track which adapter is loaded
        self.adapted_models = {}  # Cache: subreddit -> PeftModel

        # Path to LoRA adapters directory
        self.adapters_dir = Path(__file__).parent / "models" / "lora_adapters"

        logger.info(f"ToxicityPredictor initialized on device: {self.device}")

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load the fine-tuned base model."""
        model_path = Path(model_path)
        logger.info(f"Loading base model from: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.base_model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.base_model.to(self.device)
        self.base_model.eval()
        logger.info("Base model loaded successfully!")

    def get_available_adapters(self):
        """
        List all available subreddit adapters.
        """
        adapters = {}
        if not self.adapters_dir.exists():
            return adapters

        for adapter_dir in self.adapters_dir.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                # Extract subreddit name from adapter directory name
                # e.g., "reddit_valorant_adapter" -> "valorant"
                name = adapter_dir.name
                if name.startswith("reddit_") and name.endswith("_adapter"):
                    subreddit = name[7:-8]  # Remove "reddit_" prefix and "_adapter" suffix
                else:
                    subreddit = name
                adapters[subreddit.lower()] = adapter_dir

        return adapters

    def load_adapter(self, subreddit):
        """
        Load a LoRA adapter for a specific subreddit.
        """
        subreddit = subreddit.lower()

        # Check if already cached
        if subreddit in self.adapted_models:
            logger.info(f"Using cached adapter for r/{subreddit}")
            self.current_adapter = subreddit
            return True

        # Find adapter
        available = self.get_available_adapters()
        if subreddit not in available:
            logger.info(f"No adapter for r/{subreddit}, using base model")
            self.current_adapter = None
            return False

        adapter_path = available[subreddit]
        logger.info(f"Loading adapter for r/{subreddit} from {adapter_path}")

        try:
            # Load PEFT model with adapter
            adapted_model = PeftModel.from_pretrained(
                self.base_model,
                str(adapter_path)
            )
            adapted_model.to(self.device)
            adapted_model.eval()

            # Cache it
            self.adapted_models[subreddit] = adapted_model
            self.current_adapter = subreddit
            logger.info(f"Adapter for r/{subreddit} loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load adapter for r/{subreddit}: {e}")
            self.current_adapter = None
            return False

    def get_model(self, subreddit=None):
        """
        Get the appropriate model for prediction.
        """
        if subreddit:
            subreddit = subreddit.lower()
            # Try to load adapter if not current
            if self.current_adapter != subreddit:
                self.load_adapter(subreddit)

            # Return adapted model if available
            if subreddit in self.adapted_models:
                return self.adapted_models[subreddit]

        return self.base_model

    def predict(self, comment, T=1.0, subreddit=None):
        """
        Predict toxicity for a single comment.
        """
        if self.base_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get appropriate model (with adapter if available)
        model = self.get_model(subreddit)

        # Tokenize with preprocessing (uses shared utility)
        encoded = tokenize_for_bert(comment, self.tokenizer, preprocess=True)

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        logits = output.logits.detach().cpu().numpy()
        prediction = int(np.argmax(logits, axis=1)[0])

        # Temperature-scaled confidence (softmax)
        exp_logits = np.exp(logits / T)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        confidence = float(probs[0][prediction])

        return prediction, confidence

