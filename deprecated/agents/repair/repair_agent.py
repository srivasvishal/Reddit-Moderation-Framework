"""
REPAIR Agent - Diagnoses failure patterns in low-confidence predictions
Uses Llama-3-8B-Instruct via Ollama for analysis
Embeds and clusters text to find new failure patterns
"""

import sys
import json
import csv
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# AI/ML imports for embedding
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers or sklearn not available")
    print("Install with: pip install sentence-transformers scikit-learn")

from config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepairAgent:
    """REPAIR Agent - Diagnoses failure patterns using Llama-3-8B-Instruct"""
    
    def __init__(self, input_file: str = None, output_dir: str = None, 
                 ollama_base_url: str = None, model_name: str = None):
        """
        Initialize REPAIR agent
        
        Args:
            input_file: Path to low_confidence_data.csv from Prediction Service
            output_dir: Directory to save patch strategy
            ollama_base_url: Base URL for Ollama API
            model_name: Name of Llama model to use
        """
        self.input_file = Path(input_file) if input_file else None
        self.output_dir = Path(output_dir) if output_dir else Config.REPAIR_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ollama_base_url = ollama_base_url or Config.OLLAMA_BASE_URL
        self.model_name = model_name or Config.OLLAMA_REPAIR_MODEL
        self.api_available = False
        
        # Initialize embedding model
        self.embedder = None
        if EMBEDDING_AVAILABLE:
            try:
                logger.info("Loading embedding model for clustering...")
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                logger.warning("Will use text-based grouping instead of semantic clustering")
        else:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers scikit-learn")
            logger.warning("Will use text-based grouping instead of semantic clustering")
        
        # Test Ollama connection and model availability
        self.model_available = False
        self.fallback_model = None
        self._test_ollama_connection()
        
        logger.info("REPAIR Agent initialized")
        logger.info(f"Ollama URL: {self.ollama_base_url}")
        logger.info(f"Model: {self.model_name}")
        if not self.model_available:
            logger.warning("Model not available. Cluster analysis will use fallback method.")
            if self.fallback_model:
                logger.info(f"Using fallback model: {self.fallback_model}")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama and check if model exists"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.api_available = True
                logger.info(f"Ollama API available at {self.ollama_base_url}")
                
                # Check if the model exists
                models_data = response.json()
                models_list = [model.get('name', '') for model in models_data.get('models', [])]
                if self.model_name not in models_list:
                    logger.warning(f"Model '{self.model_name}' not found in Ollama.")
                    logger.warning(f"Available models: {', '.join(models_list) if models_list else 'None'}")
                    logger.warning(f"Install the model with: ollama pull {self.model_name}")
                    self.model_available = False
                    
                    # Try to find a fallback model (prefer Mistral if available)
                    fallback_options = ['mistral:7b-instruct', 'mistral', 'llama3', 'llama2']
                    for fallback in fallback_options:
                        for available_model in models_list:
                            if fallback in available_model.lower():
                                self.fallback_model = available_model
                                logger.info(f"Found fallback model: {self.fallback_model}")
                                break
                        if self.fallback_model:
                            break
                else:
                    self.model_available = True
                    logger.info(f"Model '{self.model_name}' is available")
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
                self.model_available = False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Ollama API: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
            logger.warning(f"Install Llama model: ollama pull {self.model_name}")
            self.model_available = False
    
    def load_low_confidence_data(self, input_file: str = None) -> List[Dict[str, Any]]:
        """
        Load low confidence predictions from CSV
        
        Args:
            input_file: Path to low_confidence_data.csv
            
        Returns:
            List of records with low confidence predictions
        """
        file_path = Path(input_file) if input_file else self.input_file
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        records = []
        logger.info(f"Loading low confidence data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        
        logger.info(f"Loaded {len(records)} low confidence records")
        return records
    
    def embed_and_cluster(self, texts: List[str], n_clusters: int = None) -> Dict[int, List[int]]:
        """
        Embed texts and cluster them to find similar failure patterns
        
        Args:
            texts: List of text strings
            n_clusters: Number of clusters (default: auto-detect)
            
        Returns:
            Dictionary mapping cluster_id to list of indices
        """
        if not self.embedder:
            logger.warning("Embedding model not available. Using text-based grouping.")
            # Use simple text-based clustering: group by first few words or length
            # This is a fallback when embeddings are not available
            clusters = defaultdict(list)
            
            # Group by text length buckets and first word (simple heuristic)
            for idx, text in enumerate(texts):
                if not text:
                    clusters[0].append(idx)
                    continue
                
                # Create a simple hash-based cluster key
                # Use text length and first few characters
                text_len = len(text)
                first_chars = text[:20].lower().strip() if len(text) > 20 else text.lower().strip()
                
                # Create cluster ID based on length bucket and first chars hash
                length_bucket = min(text_len // 50, 9)  # 10 buckets (0-9)
                char_hash = hash(first_chars) % 5  # 5 sub-buckets
                cluster_id = length_bucket * 5 + char_hash
                
                clusters[cluster_id].append(idx)
            
            # Limit to reasonable number of clusters (max 10)
            if len(clusters) > 10:
                # Merge smaller clusters
                sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
                merged_clusters = {}
                for i, (cluster_id, indices) in enumerate(sorted_clusters[:10]):
                    merged_clusters[i] = indices
                # Add remaining to cluster 0
                for cluster_id, indices in sorted_clusters[10:]:
                    merged_clusters[0] = merged_clusters.get(0, []) + indices
                clusters = merged_clusters
            
            logger.info(f"Created {len(clusters)} clusters using text-based grouping")
            for cluster_id, indices in clusters.items():
                logger.info(f"  Cluster {cluster_id}: {len(indices)} texts")
            
            return dict(clusters)
        
        if not texts:
            return {}
        
        logger.info(f"Embedding {len(texts)} texts...")
        
        # Generate embeddings
        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings: shape {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {0: list(range(len(texts)))}
        
        # Determine number of clusters
        if n_clusters is None:
            # Auto-detect: use sqrt of number of samples, but at least 2 and at most 10
            n_clusters = max(2, min(10, int(np.sqrt(len(texts)))))
        
        logger.info(f"Clustering into {n_clusters} clusters...")
        
        # Perform clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group indices by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[int(label)].append(idx)
            
            logger.info(f"Created {len(clusters)} clusters")
            for cluster_id, indices in clusters.items():
                logger.info(f"  Cluster {cluster_id}: {len(indices)} texts")
            
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error clustering: {e}")
            return {0: list(range(len(texts)))}
    
    def analyze_cluster_with_llama(self, comments: List[str]) -> str:
        """
        Use Llama-3-8B-Instruct to analyze a cluster of similar comments
        
        Args:
            comments: List of comment texts from the cluster
            
        Returns:
            Analysis string from Llama
        """
        if not self.api_available:
            logger.warning("Ollama API not available. Returning placeholder analysis.")
            return "Analysis: Ollama API not available. Cannot analyze cluster."
        
        if not self.model_available:
            # Try to use fallback model if available
            if self.fallback_model:
                logger.info(f"Using fallback model '{self.fallback_model}' for cluster analysis.")
                # Use fallback model for analysis
                comments_text = "\n".join([f"- {comment[:200]}" for comment in comments[:10]])
                
                prompt = f"""My toxicity model failed on these comments (confidence < 80%). They seem similar. What is the shared theme or reason for failure?

Comments:
{comments_text}

Analysis:"""
                
                try:
                    response = requests.post(
                        f"{self.ollama_base_url}/api/generate",
                        json={
                            "model": self.fallback_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": 500
                            }
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        analysis = result.get('response', '').strip()
                        if analysis.startswith('Analysis:'):
                            analysis = analysis[len('Analysis:'):].strip()
                        logger.debug(f"Fallback model analysis: {analysis[:100]}...")
                        return analysis
                    else:
                        logger.warning(f"Fallback model API returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error using fallback model: {e}")
            
            # If fallback also failed, generate basic analysis
            logger.info("Generating basic pattern-based analysis.")
            all_text = " ".join(comments[:5])
            word_count = len(all_text.split())
            avg_length = sum(len(c) for c in comments[:5]) / len(comments[:5]) if comments else 0
            
            analysis = f"Cluster contains {len(comments)} comments. "
            analysis += f"Average length: {avg_length:.0f} characters. "
            analysis += f"Common patterns detected in text. "
            if self.fallback_model:
                analysis += f"Primary model '{self.model_name}' unavailable, fallback '{self.fallback_model}' also failed."
            else:
                analysis += f"Model analysis unavailable - install '{self.model_name}' with: ollama pull {self.model_name}"
            return analysis
        
        # Format comments for prompt
        comments_text = "\n".join([f"- {comment[:200]}" for comment in comments[:10]])  # Limit to 10 comments, 200 chars each
        
        prompt = f"""My toxicity model failed on these comments (confidence < 80%). They seem similar. What is the shared theme or reason for failure?

Comments:
{comments_text}

Analysis:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more focused analysis
                        "top_p": 0.9,
                        "num_predict": 500  # Allow longer analysis
                    }
                },
                timeout=60  # 60 second timeout for analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', '').strip()
                
                # Clean up the response
                if analysis.startswith('Analysis:'):
                    analysis = analysis[len('Analysis:'):].strip()
                
                logger.debug(f"Llama analysis: {analysis[:100]}...")
                return analysis
            else:
                logger.error(f"Ollama API returned status {response.status_code}: {response.text}")
                return f"Error: API returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("Timeout calling Ollama API (60s limit exceeded)")
            return "Analysis: Timeout - analysis took too long."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Analysis: Error - {str(e)}"
    
    def load_cleaned_data(self, input_file: str = None) -> List[Dict[str, Any]]:
        """
        Load cleaned data from TRIAGE agent CSV
        
        Args:
            input_file: Path to cleaned CSV file
            
        Returns:
            List of records
        """
        file_path = Path(input_file) if input_file else self.input_file
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading cleaned data from {file_path}")
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        
        logger.info(f"Loaded {len(records)} records from cleaned dataset")
        return records
    
    def optimize_hyperparameters(self, num_samples: int, num_clusters: int) -> Dict[str, Any]:
        """
        Optimize hyperparameters for LoRA fine-tuning based on data characteristics
        
        Args:
            num_samples: Number of training samples
            num_clusters: Number of failure pattern clusters
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        logger.info("Optimizing hyperparameters for LoRA fine-tuning...")
        
        # Base hyperparameters
        base_lora_r = 8
        base_lora_alpha = 16
        base_batch_size = 8
        base_learning_rate = 2e-4
        base_num_epochs = 3
        
        # Adjust based on data size
        if num_samples < 100:
            lora_r = 4
            lora_alpha = 8
            batch_size = 4
            learning_rate = 5e-4
            num_epochs = 5
        elif num_samples < 500:
            lora_r = 8
            lora_alpha = 16
            batch_size = 8
            learning_rate = 2e-4
            num_epochs = 3
        elif num_samples < 2000:
            lora_r = 16
            lora_alpha = 32
            batch_size = 16
            learning_rate = 1e-4
            num_epochs = 3
        else:
            lora_r = 32
            lora_alpha = 64
            batch_size = 32
            learning_rate = 5e-5
            num_epochs = 2
        
        # Adjust based on number of clusters (more clusters = more complex patterns)
        if num_clusters > 5:
            lora_r = min(lora_r * 2, 64)  # Increase capacity for complex patterns
            lora_alpha = min(lora_alpha * 2, 128)
        
        hyperparameters = {
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': 0.1,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'warmup_steps': max(10, num_samples // 20),
            'weight_decay': 0.01,
            'max_seq_length': 512,
            'gradient_accumulation_steps': 1
        }
        
        logger.info(f"Optimized hyperparameters: r={lora_r}, alpha={lora_alpha}, "
                   f"batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
        
        return hyperparameters
    
    def create_adapt_config(self, hyperparameters: Dict[str, Any], 
                           patch_strategy_path: str,
                           baseline_model_gdrive_link: str = None,
                           cleaned_data_file: str = None) -> str:
        """
        Create configuration file for ADAPT agent with hyperparameters and model link
        
        Args:
            hyperparameters: Optimized hyperparameters
            patch_strategy_path: Path to patch strategy JSON
            baseline_model_gdrive_link: Google Drive link to baseline model (or "false" for local mode)
            cleaned_data_file: Path to cleaned data file from TRIAGE (for local training)
            
        Returns:
            Path to config file
        """
        # If baseline_model_gdrive_link is "false" or not set, use local mode
        if not baseline_model_gdrive_link or str(baseline_model_gdrive_link).lower() == 'false':
            baseline_model_gdrive_link = "false"  # Explicitly set to false for local mode
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = self.output_dir / f"adapt_config_{timestamp}.json"
        
        adapt_config = {
            'timestamp': datetime.now().isoformat(),
            'baseline_model_gdrive_link': baseline_model_gdrive_link,
            'patch_strategy_file': patch_strategy_path,
            'cleaned_data_file': cleaned_data_file,  # Include cleaned data file for local training
            'hyperparameters': hyperparameters,
            'lora_config': {
                'r': hyperparameters['lora_r'],
                'alpha': hyperparameters['lora_alpha'],
                'dropout': hyperparameters['lora_dropout'],
                'target_modules': ['query', 'value', 'key', 'dense'],
                'task_type': 'SEQ_CLS'
            },
            'training_config': {
                'batch_size': hyperparameters['batch_size'],
                'learning_rate': hyperparameters['learning_rate'],
                'num_epochs': hyperparameters['num_epochs'],
                'warmup_steps': hyperparameters['warmup_steps'],
                'weight_decay': hyperparameters['weight_decay'],
                'max_seq_length': hyperparameters['max_seq_length'],
                'gradient_accumulation_steps': hyperparameters['gradient_accumulation_steps'],
                'save_strategy': 'epoch',
                'evaluation_strategy': 'no',
                'logging_steps': 10
            },
            'device': 'auto',  # Will be auto-detected
            'output_dir': str(Config.ADAPT_OUTPUT_DIR)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(adapt_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ADAPT config file created: {config_path}")
        return str(config_path)
    
    def process_all(self, output_filename: str = None, 
                   use_cleaned_data: bool = False,
                   baseline_model_gdrive_link: str = None) -> tuple:
        """
        Process cleaned data from TRIAGE and generate patch strategy + ADAPT config
        
        Args:
            output_filename: Output filename for patch strategy
            use_cleaned_data: If True, use cleaned data from TRIAGE; if False, use low_confidence data
            baseline_model_gdrive_link: Google Drive link to baseline model
            
        Returns:
            tuple: (patch_strategy_path, adapt_config_path)
        """
        if not self.input_file:
            raise ValueError("Input file not specified")
        
        logger.info("="*60)
        logger.info("Starting REPAIR Agent processing...")
        logger.info("="*60)
        
        # Load data (either cleaned from TRIAGE or low_confidence from Prediction)
        if use_cleaned_data:
            records = self.load_cleaned_data()
        else:
            records = self.load_low_confidence_data()
        
        if not records:
            logger.warning("No records to process")
            return None, None
        
        # Extract comment texts
        texts = []
        for record in records:
            comment_text = record.get('comment_body') or record.get('comments') or record.get('text', '')
            if comment_text:
                texts.append(comment_text)
        
        if not texts:
            logger.warning("No comment texts found in records")
            return None, None
        
        # Embed and cluster
        clusters = self.embed_and_cluster(texts)
        
        # Analyze each cluster
        cluster_analyses = {}
        cluster_data = {}
        
        for cluster_id, indices in clusters.items():
            logger.info(f"\nAnalyzing cluster {cluster_id} ({len(indices)} comments)...")
            
            # Get comments for this cluster
            cluster_comments = [texts[i] for i in indices]
            
            # Analyze with Llama
            analysis = self.analyze_cluster_with_llama(cluster_comments)
            cluster_analyses[cluster_id] = analysis
            
            # Store cluster data
            cluster_data[cluster_id] = {
                'indices': indices,
                'comments': cluster_comments[:5],  # Store first 5 as examples
                'analysis': analysis,
                'count': len(indices)
            }
            
            logger.info(f"Cluster {cluster_id} analysis: {analysis[:100]}...")
        
        # Optimize hyperparameters
        hyperparameters = self.optimize_hyperparameters(len(records), len(clusters))
        
        # Generate patch strategy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"patch_strategy_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        # Create patch strategy
        patch_strategy = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(self.input_file),
            'total_records': len(records),
            'clusters': len(clusters),
            'cluster_analyses': cluster_analyses,
            'cluster_data': {
                str(k): {
                    'count': v['count'],
                    'sample_comments': v['comments'],
                    'analysis': v['analysis'],
                    'data_path': str(self.input_file)
                }
                for k, v in cluster_data.items()
            },
            'optimized_hyperparameters': hyperparameters
        }
        
        # Save patch strategy
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patch_strategy, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nPatch strategy saved to: {output_path}")
        logger.info(f"Total clusters analyzed: {len(clusters)}")
        logger.info(f"Total records: {len(records)}")
        
        # Create ADAPT config file
        # Pass the input file path as cleaned_data_file for local training
        cleaned_data_file = str(self.input_file) if self.input_file else None
        adapt_config_path = self.create_adapt_config(
            hyperparameters,
            str(output_path),
            baseline_model_gdrive_link,
            cleaned_data_file
        )
        
        return str(output_path), adapt_config_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='REPAIR Agent - Diagnose failure patterns in low-confidence predictions'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to cleaned CSV from TRIAGE agent or low_confidence_data.csv from Prediction Service'
    )
    
    parser.add_argument(
        '--use-cleaned-data',
        action='store_true',
        help='Use cleaned data from TRIAGE agent (default: False, uses low_confidence data)'
    )
    
    parser.add_argument(
        '--baseline-model-gdrive',
        type=str,
        default=None,
        help='Google Drive link to baseline model (default: placeholder link)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output filename (default: auto-generated)'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default=None,
        help='Ollama base URL (default: from config)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Llama model name (default: llama3:8b-instruct)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=None,
        help='Number of clusters (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("REPAIR Agent - Failure Pattern Diagnosis")
    print("="*70)
    print(f"Input file: {args.input_file}")
    print(f"Ollama URL: {args.ollama_url or Config.OLLAMA_BASE_URL}")
    print(f"Model: {args.model or Config.OLLAMA_REPAIR_MODEL}")
    print("="*70)
    
    try:
        agent = RepairAgent(
            input_file=args.input_file,
            output_dir=args.output_dir,
            ollama_base_url=args.ollama_url,
            model_name=args.model
        )
        
        patch_strategy_path, adapt_config_path = agent.process_all(
            output_filename=args.output_file,
            use_cleaned_data=args.use_cleaned_data,
            baseline_model_gdrive_link=args.baseline_model_gdrive
        )
        
        if patch_strategy_path:
            print(f"\n✓ Patch strategy saved to: {patch_strategy_path}")
            if adapt_config_path:
                print(f"✓ ADAPT config saved to: {adapt_config_path}")
                print("Ready for ADAPT agent to apply patches.")
            else:
                print("⚠ ADAPT config not generated")
        else:
            print("\n⚠ No output generated")
            
    except Exception as e:
        logger.error(f"REPAIR agent failed: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
