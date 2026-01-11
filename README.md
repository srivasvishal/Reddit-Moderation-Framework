# Reddit Moderation Framework

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Required packages installed (see `requirements.txt`)
- Ollama running (optional, for triage preprocessing)

### Running the Dashboard

Follow these steps in order:

#### 1. Start the Reddit Proxy Server
First, launch the proxy server that handles Reddit traffic and toxicity detection:

```bash
python reddit_proxy.py
```

The proxy server runs on `http://localhost:5000` and:
- Proxies old.reddit.com traffic
- Runs toxicity predictions on comments using BERT
- Applies subreddit-specific LoRA adapters when available
- Injects toxicity overlay UI into Reddit pages

#### 2. Start Ollama (Optional)
For triage preprocessing (slang expansion, acronym detection, censored word handling):

```bash
ollama serve
ollama pull ministral-3:3b
```

#### 3. Launch the Streamlit Dashboard
In a new terminal, start the dashboard:

```bash
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501` and displays:
- Embedded Reddit browser with toxicity overlays
- Real-time triage processing progress
- Overview metrics and fix type analysis
- Processing timeline statistics

## Project Structure

| File | Description |
|------|-------------|
| `reddit_proxy.py` | Flask proxy server for Reddit with toxicity injection |
| `dashboard.py` | Streamlit dashboard for moderation UI |
| `toxicity_predictor.py` | BERT-based toxicity prediction with LoRA adapter support |
| `adapt_agent.py` | LoRA adapter training using PEFT |
| `triage_agent.py` | AI-powered text preprocessing (slang/acronym expansion) |
| `text_processing.py` | Text tokenization utilities for BERT |
| `train_subreddit_adapters.py` | Script to train subreddit-specific adapters |

## Training Subreddit Adapters

Train LoRA adapters from human feedback collected via the dashboard:

```bash
python train_subreddit_adapters.py --feedback-csv dashboard_data/human_feedback.csv
```

Options:
- `--min-samples`: Minimum feedback samples required (default: 300)
- `--epochs`: Training epochs (default: 3)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--validation-split`: Train/validation split (default: 0.2)
- `--use-triage`: Enable triage preprocessing

Monitor training with TensorBoard:
```bash
tensorboard --logdir=runs
```

## Models

Place your fine-tuned BERT model at:
```
models/bert_model_full_data (This is the fine-tuned BERT model we used for the project)
```

Trained LoRA adapters are saved to:
```
models/lora_adapters/reddit_<subreddit>_adapter/
```

## How It Works

1. **Proxy Server**: Intercepts Reddit pages and extracts comments
2. **Triage Agent**: Preprocesses text (expands slang, de-obfuscates censored words)
3. **Toxicity Predictor**: Classifies comments as toxic/non-toxic using BERT
4. **LoRA Adapters**: Subreddit-specific fine-tuning from human feedback
5. **Dashboard**: Displays results with confidence badges and feedback buttons

## Providing Feedback

On any Reddit page through the proxy:
- Comments are highlighted (green=safe, red=toxic, yellow=uncertain)
- Click "Toxic" or "Safe" buttons to provide feedback
- Feedback is saved to `dashboard_data/human_feedback.csv`
- Use collected feedback to train subreddit-specific adapters
