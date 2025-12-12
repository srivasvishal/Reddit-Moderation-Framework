import re
import torch


# TOKENIZATION CONSTANTS
# =============================================================================
MAX_TOKEN_LENGTH = 588          # Initial tokenization length before truncation
HEAD_TOKENS = 128               # First N tokens to keep (captures opening/context)
TAIL_TOKENS = 382               # Last N tokens to keep (captures conclusion)
FINAL_SEQUENCE_LENGTH = 510     # HEAD_TOKENS + TAIL_TOKENS = 128 + 382 = 510
TOXICITY_THRESHOLD = 0.5        # Score >= this is classified as toxic


def preprocess_text(comment: str) -> str:
    """
    Preprocess text for BERT toxicity classification.
    """
    original = comment
    # Remove leading/trailing punctuation
    comment = re.sub(r'^[\(\)\'\"\.\!\?]+|[\)\'\"\.\!\?]+$', ' ', comment)
    # Remove quotes
    comment = re.sub(r'\'|\"', '', comment)
    # Remove special characters around spaces and punctuation
    comment = re.sub(r'[^a-zA-Z0-9]\s|\s[^a-zA-Z0-9]|\-|\(|\)|\?|\!|—|\/', ' ', comment)
    # Normalize whitespace and lowercase
    comment = re.sub(r'[\s\.]+', ' ', comment).strip().lower()

    return comment if comment else original


def tokenize_for_bert(text: str, tokenizer, preprocess: bool = True) -> dict:
    """
    Tokenize text using BERT tokenizer with head+tail concatenation.

    Uses the 588 -> 510 token strategy: keeps first 128 and last 382 tokens
    to capture both context (beginning) and conclusion (end) of long texts.
    """
    if preprocess:
        text = preprocess_text(text)

    # Tokenize to MAX_TOKEN_LENGTH
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LENGTH,
        truncation=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Concatenate: first HEAD_TOKENS + last TAIL_TOKENS
    input_ids = torch.cat((
        encoded['input_ids'][:, :HEAD_TOKENS],
        encoded['input_ids'][:, -TAIL_TOKENS:]
    ), dim=1)

    attention_mask = torch.cat((
        encoded['attention_mask'][:, :HEAD_TOKENS],
        encoded['attention_mask'][:, -TAIL_TOKENS:]
    ), dim=1)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
