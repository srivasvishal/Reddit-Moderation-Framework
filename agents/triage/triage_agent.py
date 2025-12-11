"""
TRIAGE AI Agent

Uses Mistral-7B-Instruct via Ollama for:
- Slang expansion (lol -> laughing out loud)
- Acronym expansion (fyi -> for your information)
- Censored word de-obfuscation 
- Intelligent data quality assessment


"""

import json
import os
import re
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
import sys
import csv


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


try:
    import os
    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['SKIP_TF_TESTS'] = '1'
    
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"Warning: Transformers not available: {e}")


from config.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device():
    
    if not AI_AVAILABLE:
        return -1, "CPU (AI not available)"
    
    try:
        
        if torch.cuda.is_available():
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            logger.info(f"GPU detected: {device_name}")
           
            return 0, device_name
        
        # Check for MPS (Apple Silicon GPU - Metal Performance Shaders)

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_name = "MPS (Apple Silicon GPU)"
            logger.info(f"GPU detected: {device_name}")

            try:
                # Test if MPS device works
                test_tensor = torch.zeros(1).to("mps")
                del test_tensor
                return torch.device("mps"), device_name
            except Exception as mps_error:
                logger.warning(f"MPS device test failed: {mps_error}. Falling back to CPU.")
                return -1, "CPU (MPS unavailable)"
        
        # Fallback to CPU
        device_name = "CPU"
        logger.info("No GPU detected, using CPU")
        return -1, device_name
        
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}. Falling back to CPU.")
        return -1, "CPU (fallback)"


class AISlangExpander:
    
    
    def __init__(self, ollama_base_url: str = None, model_name: str = "mistral:7b-instruct"):
        """
        Initialize the AI Slang Expander with Ollama
        
        Args:
            ollama_base_url: Base URL for Ollama API (default: http://localhost:11434)
            model_name: Name of the model to use (default: mistral:7b-instruct)
        """
        self.ollama_base_url = ollama_base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'mistral:7b-instruct')
        self.api_available = False
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.api_available = True
                logger.info(f"Ollama API available at {self.ollama_base_url}")
                logger.info(f"Using model: {self.model_name}")
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Ollama API at {self.ollama_base_url}: {e}")
            logger.warning("Make sure Ollama is running: ollama serve")
            logger.warning(f"Or install Mistral model: ollama pull {self.model_name}")
    
    def expand_slang_ai(self, text: str) -> tuple:
        """
        Use Mistral-7B-Instruct via Ollama to:
        1. Identify abbreviations, slang, and censored words that need expansion
        2. Expand/replace them all in one call
        3. Return both cleaned text and list of changes
        
        Returns: (expanded_text, fixes_applied_list)
        """
        if not text:
            return text, []
        
        if not self.api_available:
            logger.warning("Ollama API not available. Returning original text.")
            return text, []
        
        try:
            # Single AI call to identify and expand everything
            cleaned_text, fixes_applied = self._call_mistral_for_cleaning_and_tracking(text)
            return cleaned_text, fixes_applied
        except Exception as e:
            logger.error(f"Error calling Mistral for text cleaning: {e}")
            return text, []
    
    def _call_mistral_for_cleaning_and_tracking(self, text: str) -> tuple:
        """
        Call Mistral-7B-Instruct via Ollama to:
        1. Identify abbreviations, slang, and censored words that need expansion
        2. Expand all slang, fix all acronyms, and de-obfuscate censored words
        3. Return both cleaned text and list of changes made
        
        Returns: (cleaned_text, list_of_fixes)
        """
        prompt = f""" Analyze the text below and:

1. IDENTIFY all abbreviations, acronyms, slang words, and censored words (with special characters like * @ # $) that need to be expanded or de-obfuscated.

2. EXPAND all identified items:
   - Expand abbreviations/acronyms to their full forms (e.g., "FYI" → "For Your Information", "ASAP" → "As Soon As Possible")
   - Expand slang to proper words (e.g., "lol" → "laugh out loud", "omg" → "oh my god")
   - De-obfuscate censored words (e.g., "f*ck" → "fuck", "@ss" → "ass")

3. Return your response in this EXACT format:
CLEANED_TEXT: [the fully cleaned and expanded text]
CHANGES:
ABBREVIATION: [original] → [expanded]
SLANG: [original] → [expanded]
CENSORED: [original] → [expanded]

If no changes are needed, return:
CLEANED_TEXT: [original text unchanged]
CHANGES: NONE

Input text:
{text}

Output:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 800
                    }
                },
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                

                cleaned_text = text
                fixes_applied = []
                

                if 'CLEANED_TEXT:' in ai_response:
                    cleaned_section = ai_response.split('CLEANED_TEXT:')[1]
                    if 'CHANGES:' in cleaned_section:
                        cleaned_text = cleaned_section.split('CHANGES:')[0].strip()
                    else:
                        cleaned_text = cleaned_section.strip()
                    

                    for prefix in ['Output:', 'The cleaned text is:', 'Here is the cleaned text:']:
                        if cleaned_text.lower().startswith(prefix.lower()):
                            cleaned_text = cleaned_text[len(prefix):].strip()
                

                if 'CHANGES:' in ai_response:
                    changes_section = ai_response.split('CHANGES:')[1].strip()
                    
                    if changes_section.upper() != 'NONE':
                        lines = changes_section.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            

                            if ':' in line and '→' in line:
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    category = parts[0].strip().upper()
                                    change_part = parts[1].strip()
                                    
                                    if '→' in change_part:
                                        change_parts = change_part.split('→', 1)
                                        original = change_parts[0].strip()
                                        expanded = change_parts[1].strip()
                                        
                                        if original and expanded:
                                            if category == 'ABBREVIATION':
                                                fixes_applied.append({
                                                    'original': original,
                                                    'expanded': expanded,
                                                    'type': 'abbreviation_expansion'
                                                })
                                            elif category == 'SLANG':
                                                fixes_applied.append({
                                                    'original': original,
                                                    'expanded': expanded,
                                                    'type': 'slang_expansion'
                                                })
                                            elif category == 'CENSORED':
                                                fixes_applied.append({
                                                    'original': original,
                                                    'expanded': expanded,
                                                    'type': 'censored_word_expansion'
                                                })
                

                if cleaned_text == text and fixes_applied == []:
                    lines = ai_response.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > len(text) * 0.5:  # Substantial line

                            for prefix in ['Output:', 'Cleaned text:', 'The cleaned text is:', 'Here is the cleaned text:', 'CLEANED_TEXT:']:
                                if line.lower().startswith(prefix.lower()):
                                    line = line[len(prefix):].strip()
                            if line and line != text:
                                cleaned_text = line
                                break
                
                return cleaned_text, fixes_applied
            else:
                logger.error(f"Ollama API returned status {response.status_code}: {response.text}")
                return text, []
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling Ollama API (45s limit exceeded)")
            return text, []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return text, []
    
    def _identify_abbreviations_with_ai_legacy(self, original: str, cleaned: str) -> dict:
        
        if original == cleaned:
            return {'abbreviations': [], 'slang': [], 'censored': []}
        
        prompt = f""" Compare the original and cleaned text below.

Original text: {original}

Cleaned text: {cleaned}

Identify ALL abbreviations, acronyms, slang words, and censored words (with special characters like * @ # $) that were in the original text and were expanded or de-obfuscated in the cleaned text.

Return ONLY a JSON list of the original words/phrases that were changed, one per line, in this exact format:
ABBREVIATION: [original word/phrase]
SLANG: [original word/phrase]
CENSORED: [original word/phrase]

If no abbreviations, slang, or censored words were found, return "NONE".

Examples:
ABBREVIATION: FYI
ABBREVIATION: ASAP
SLANG: lol
CENSORED: f*ck
CENSORED: @ss

Output:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 200
                    }
                },
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                

                abbreviations = []
                slang = []
                censored = []
                
                lines = ai_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or line.upper() == 'NONE':
                        continue
                    

                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            category = parts[0].strip().upper()
                            word = parts[1].strip()
                            
                            if category == 'ABBREVIATION' and word:
                                abbreviations.append(word)
                            elif category == 'SLANG' and word:
                                slang.append(word)
                            elif category == 'CENSORED' and word:
                                censored.append(word)
                
                return {
                    'abbreviations': abbreviations,
                    'slang': slang,
                    'censored': censored
                }
            else:
                logger.warning(f"Ollama API returned status {response.status_code} for abbreviation detection: {response.text}")
                return {'abbreviations': [], 'slang': [], 'censored': []}
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout calling Ollama API for abbreviation detection (20s limit exceeded)")
            return {'abbreviations': [], 'slang': [], 'censored': []}
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error calling Ollama API for abbreviation detection: {e}")
            return {'abbreviations': [], 'slang': [], 'censored': []}
    
    def _extract_changes(self, original: str, cleaned: str, ai_fixes: list = None) -> list:

        import re
        
        fixes = []
        
        if original == cleaned:
            return fixes

        if ai_fixes:
            common_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
                'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
                'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
                'did', 'let', 'put', 'say', 'she', 'too', 'use', 'had', 'got', 'is',
                'it', 'to', 'of', 'in', 'on', 'at', 'as', 'be', 'we', 'if', 'or',
                'an', 'do', 'my', 'so', 'up', 'am', 'go', 'no', 'me', 'he', 'by',
                'us', 'oh', 'hi', 'ok', 'even', 'they', 'fix', 'what', 'which',
                'feels', 'cold', 'other', 'feel', 're', 'alone', 'many', 'very',
                'act', 'human', 'with', 'this', 'that', 'from', 'have', 'been',
                'will', 'would', 'could', 'should', 'when', 'where', 'why', 'how'
            }
            
            seen = set()
            for fix in ai_fixes:

                if fix.get('original', '').lower() == fix.get('expanded', '').lower():
                    continue
                

                if fix.get('original', '').lower() in common_words:
                    continue
                
                key = (fix.get('original', '').lower(), fix.get('expanded', '').lower())
                if key not in seen:
                    seen.add(key)
                    fixes.append(fix)
            
            return fixes
        

        censored_pattern = r'\b([a-zA-Z]*[*@#$%_0-9]+[a-zA-Z]*)\b'
        for match in re.finditer(censored_pattern, original):
            censored = match.group(1)
            censored_lower = censored.lower()
            

            if censored_lower not in cleaned.lower():

                censored_clean = re.sub(r'[*@#$%_0-9]', '', censored)
                words_in_cleaned = re.findall(r'\b\w+\b', cleaned)
                for word in words_in_cleaned:
                    if word.lower().startswith(censored_clean.lower()[:2]) and \
                       len(word) >= len(censored_clean):
                        fixes.append({
                            'original': censored,
                            'expanded': word,
                            'type': 'censored_word_expansion'
                        })
                        break
        
        return fixes
    


class AIDataQualityChecker:

    
    def __init__(self):
        self.classifier = None
        if AI_AVAILABLE:
            try:

                model_name = "distilbert-base-uncased-finetuned-sst-2-english"

                device, device_name = get_device()
                logger.info(f"Loading quality checker on {device_name}")
                
                # Handle MPS device compatibility
                try:
                    if isinstance(device, torch.device) and device.type == "mps":
                        # For MPS
                        try:
                            self.classifier = pipeline(
                                "sentiment-analysis",
                                model=model_name,
                                device=device
                            )
                        except (ValueError, TypeError, RuntimeError) as mps_error:

                            logger.warning(f"MPS not supported in pipeline: {mps_error}. Using CPU instead.")
                            self.classifier = pipeline(
                                "sentiment-analysis",
                                model=model_name,
                                device=-1
                            )
                    else:
                        # CUDA or CPU - standard handling
                        self.classifier = pipeline(
                            "sentiment-analysis",
                            model=model_name,
                            device=device
                        )
                except Exception as pipeline_error:
                    logger.warning(f"Pipeline creation failed: {pipeline_error}. Falling back to CPU.")
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                        device=-1
                )
                logger.info(f"AI quality checker loaded on {device_name}")
            except Exception as e:
                logger.warning(f"Could not load AI quality checker: {e}")
    
    def check_quality_ai(self, text: str) -> Dict[str, Any]:

        issues = []
        
        if not text:
            return {'has_issues': True, 'issues': ['empty_content']}
        

        if self.classifier:
            try:

                result = self.classifier(text[:512])
                score = result[0]['score'] if result else 0.5
                
                if score < 0.3:
                    issues.append({
                        'type': 'low_quality_signal',
                        'confidence': score,
                        'severity': 'medium'
                    })
            except Exception as e:
                logger.warning(f"AI quality check failed: {e}")
        
        # Standard checks
        if len(text) < 10:
            issues.append({'type': 'very_short', 'severity': 'low'})
        if '�' in text:
            issues.append({'type': 'encoding_issue', 'severity': 'high'})
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues
        }


class FixTracker:

    
    def __init__(self):
        self.fixes = defaultdict(list)
        self.stats = {
            'total_fixes': 0,
            'fixes_by_type': defaultdict(int),
            'records_fixed': 0
        }
    
    def record_fix(self, record_id: str, fix_type: str, details: Dict[str, Any]):

        fix_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': fix_type,
            'details': details
        }
        self.fixes[record_id].append(fix_entry)
        self.stats['total_fixes'] += 1
        self.stats['fixes_by_type'][fix_type] += 1
    
    def get_fixes_for_record(self, record_id: str) -> List[Dict]:

        return self.fixes.get(record_id, [])
    
    def get_statistics(self) -> Dict:

        return dict(self.stats)


class StatisticsRecorder:
    """Record processing statistics"""
    
    def __init__(self):
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_records': 0,
            'processed_records': 0,
            'records_with_issues': 0,
            'records_fixed': 0,
            'records_dropped': 0,
            'ai_expansions': 0,
            'rule_based_expansions': 0,
            'processing_time_seconds': 0
        }
    
    def increment(self, key: str, value: int = 1):
        """Increment a statistic"""
        if key in self.stats:
            self.stats[key] += value
    
    def set(self, key: str, value):
        """Set a statistic"""
        if key in self.stats:
            self.stats[key] = value
    
    def get_stats(self) -> Dict:

        self.stats['end_time'] = datetime.now().isoformat()
        return self.stats
    
    def save_to_file(self, filepath: str):

        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)


class TriageAIAgent:
    """Main TRIAGE AI Agent class"""
    
    def __init__(self, api_url: str = None, input_file: str = None, output_dir: str = 'data/cleaned', limit: int = None):

        self.api_url = api_url or os.getenv('API_URL', 'http://localhost:5001')
        self.input_file = Path(input_file) if input_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        

        self.ai_expander = AISlangExpander()
        self.ai_checker = AIDataQualityChecker()
        

        self.fix_tracker = FixTracker()
        self.stats_recorder = StatisticsRecorder()
        

        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / f"triage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        logger.info("TRIAGE AI Agent initialized")
        logger.info(f"AI Available: {AI_AVAILABLE}")
        if self.input_file:
            logger.info(f"Input file: {self.input_file}")
        else:
            logger.info(f"API URL: {self.api_url}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data_from_file(self) -> List[Dict[str, Any]]:
        """
        Load data from a local file (CSV, JSON, or JSONL format)
        
        Returns:
            List of records as dictionaries
        """
        if not self.input_file or not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        records = []
        file_ext = self.input_file.suffix.lower()
        
        try:
            if file_ext == '.csv':
                logger.info(f"Loading data from CSV file: {self.input_file}")
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        if self.limit and len(records) >= self.limit:
                            logger.info(f"Reached limit of {self.limit} records, stopping load")
                            break
                        
                        # Parse JSON strings in CSV cells
                        record = {}
                        for key, value in row.items():
                            if not value:
                                record[key] = None
                            elif value.startswith('{') or value.startswith('['):
                                try:
                                    record[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    record[key] = value
                            else:
                                record[key] = value
                        
                        if record:
                            records.append(record)
            
            elif file_ext in ['.json', '.jsonl']:
                logger.info(f"Loading data from {'JSONL' if file_ext == '.jsonl' else 'JSON'} file: {self.input_file}")
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    if file_ext == '.jsonl':
                        # JSONL: one JSON object per line
                        for line_num, line in enumerate(f, 1):
                            if self.limit and len(records) >= self.limit:
                                logger.info(f"Reached limit of {self.limit} records, stopping load")
                                break
                            
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                record = json.loads(line)
                                if isinstance(record, dict):
                                    records.append(record)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    else:
                        # JSON: single array or object
                        data = json.load(f)
                        if isinstance(data, list):
                            records = data[:self.limit] if self.limit else data
                        elif isinstance(data, dict):
                            records = [data]
                        else:
                            raise ValueError(f"JSON file must contain an array or object, got {type(data)}")
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .csv, .json, .jsonl")
            
            logger.info(f"Loaded {len(records)} records from {self.input_file}" + (f" (limited to {self.limit})" if self.limit else ""))
            self.stats_recorder.set('total_records', len(records))
            return records
        
        except Exception as e:
            logger.error(f"Error loading data from file {self.input_file}: {e}")
            raise
    
    def fetch_data(self) -> List[Dict[str, Any]]:

        try:
            limit_msg = f" (limit: {self.limit})" if self.limit else ""
            logger.info(f"Fetching data from {self.api_url}/api/records{limit_msg}")
            response = requests.get(f"{self.api_url}/api/records", timeout=30)
            response.raise_for_status()
            
            records = []

            csv_content = response.text
            csv_reader = csv.DictReader(csv_content.splitlines())
            
            for row in csv_reader:

                if self.limit and len(records) >= self.limit:
                    logger.info(f"Reached limit of {self.limit} records, stopping fetch")
                    break
                

                record = {}
                for key, value in row.items():
                    if not value:
                        record[key] = None
                    elif value.startswith('{') or value.startswith('['):

                        try:
                            record[key] = json.loads(value)
                        except json.JSONDecodeError:
                            record[key] = value
                    else:
                        record[key] = value
                
                if record:
                            records.append(record)
            
            logger.info(f"Fetched {len(records)} records" + (f" (limited to {self.limit})" if self.limit else ""))
            self.stats_recorder.set('total_records', len(records))
            return records
        
        except requests.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def check_record(self, record: Dict[str, Any]) -> Dict[str, Any]:

        import re
        

        record_id = record.get('comment_id') or record.get('url') or f"record_{hash(str(record))}"
        

        issues = []
        

        comment_text = record.get('comment_body') or record.get('comments') or record.get('comment_text', '')
        
        if not comment_text or comment_text.strip() == '':
            issues.append({'type': 'missing_comment_body', 'severity': 'high'})
            return {
                'has_issues': True,
                'issues': issues,
                'record_id': record_id
            }
        

        detected_patterns = []
        

        common_abbrevs = ['fyi', 'asap', 'lol', 'omg', 'tbh', 'idk', 'imo', 'btw', 'wtf', 'lmao', 'ur', 'u', 'r', 'thx', 'ty', 'np', 'fr', 'ngl', 'nvm', 'tldr']
        found_abbrevs = []
        for abbrev in common_abbrevs:
            if re.search(r'\b' + re.escape(abbrev) + r'\b', comment_text, re.IGNORECASE):
                found_abbrevs.append(abbrev.upper())
        
        if found_abbrevs:
            detected_patterns.append(f"abbreviations_found: {', '.join(found_abbrevs)}")
            issues.append({
                'type': 'abbreviations_detected',
                'severity': 'low',
                'details': {'abbreviations': found_abbrevs, 'count': len(found_abbrevs)}
            })
        

        censored_patterns = [
            r'f\*+ck', r'sh\*+t', r'b\*+tch', r'a\*+s', r'@ss', r'@\*+ss',
            r'd\*+mn', r'h\*+ll', r'cr\*+p', r'd\*+ck', r'p\*+ss', r'c\*+nt'
        ]
        found_censored = []
        for pattern in censored_patterns:
            if re.search(pattern, comment_text, re.IGNORECASE):
                found_censored.append(pattern.replace('\\', '').replace('+', '*'))
        
        if found_censored:
            detected_patterns.append(f"censored_words_found: {', '.join(found_censored[:3])}")  # Limit to 3
            issues.append({
                'type': 'censored_words_detected',
                'severity': 'medium',
                'details': {'patterns': found_censored, 'count': len(found_censored)}
            })
        

        mixed_char_pattern = r'\b\w*[a-zA-Z]\w*[0-9@#$%*]\w*[a-zA-Z]\w*\b'
        if re.search(mixed_char_pattern, comment_text):
            detected_patterns.append("mixed_characters_detected")
            issues.append({
                'type': 'mixed_characters',
                'severity': 'low',
                'details': 'Text contains mixed alphanumeric/special character patterns'
            })
        

        short_words = re.findall(r'\b[a-zA-Z]{2,3}\b', comment_text)
        if len(short_words) > 3:
            detected_patterns.append(f"multiple_short_forms: {len(short_words)} short words")
            issues.append({
                'type': 'multiple_short_forms',
                'severity': 'low',
                'details': {'short_word_count': len(short_words)}
            })
        

        ai_quality = self.ai_checker.check_quality_ai(comment_text)
        
        if ai_quality['has_issues']:
            issues.extend(ai_quality['issues'])
            self.stats_recorder.increment('records_with_issues')
        

        if detected_patterns:
            issues.append({
                'type': 'detected_patterns',
                'severity': 'info',
                'details': detected_patterns
            })
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'record_id': record_id
        }
    
    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:

        record_id = record.get('comment_id') or record.get('url') or f"record_{hash(str(record))}"
        cleaned = record.copy()
        

        score_field = 'comment_score' if 'comment_score' in cleaned else 'score'
        if score_field in cleaned:
            try:
                cleaned[score_field] = int(cleaned[score_field])
            except (ValueError, TypeError):
                cleaned[score_field] = 0
                self.fix_tracker.record_fix(record_id, 'type_fix', {'field': score_field, 'value': 0})
        

        # Try to find the comment field - check multiple possible field names
        comment_field = None
        for field_name in ['comment_body', 'comments', 'comment_text']:
            if field_name in cleaned and cleaned[field_name]:
                comment_field = field_name
                break
        
        if comment_field and cleaned[comment_field]:
            original_comment = cleaned[comment_field]
            

            expanded_comment, fixes_applied = self.ai_expander.expand_slang_ai(original_comment)
            cleaned[comment_field] = expanded_comment
            

            cleaned['_fixes_applied_list'] = fixes_applied
            
            if original_comment != expanded_comment:
                cleaned[f'_original_{comment_field}'] = original_comment
                cleaned['_slang_expanded'] = True
                

                for fix in fixes_applied:
                    self.fix_tracker.record_fix(
                        record_id,
                        fix.get('type', 'slang_expansion'),
                        fix
                    )
                    self.stats_recorder.increment('ai_expansions')
                
                logger.debug(f"Expanded slang for record {record_id}: {len(fixes_applied)} fixes applied")
            else:
                cleaned['_fixes_applied_list'] = []
                logger.debug(f"No slang expansion needed for record {record_id}")
        

        if comment_field and comment_field in cleaned:
            cleaned[comment_field] = cleaned[comment_field].replace('�', '')
        

        cleaned['_triage_timestamp'] = datetime.now().isoformat()
        cleaned['_triage_status'] = 'cleaned'
        cleaned['_fixes_applied'] = len(self.fix_tracker.get_fixes_for_record(record_id))
        
        self.stats_recorder.increment('records_fixed')
        return cleaned
    
    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        start_time = datetime.now()
        

        if not isinstance(record, dict):
            logger.error(f"Record is not a dictionary: {type(record)}")
            return None
        
        try:

            check_result = self.check_record(record)
            
            if check_result['has_issues']:
                critical_issues = [i for i in check_result['issues'] 
                                  if i.get('severity') == 'high']
                
                if critical_issues and 'missing_comment_body' in [i['type'] for i in critical_issues]:
                    logger.warning(f"Dropping record with missing comment body: {check_result['record_id']}")
                    self.stats_recorder.increment('records_dropped')
                    return None
            

            cleaned = self.clean_record(record)
            

            quality_issues_details = []
            

            if check_result['has_issues']:
                for issue in check_result['issues']:
                    issue_detail = {
                        'issue_type': issue.get('type', 'unknown'),
                        'description': self._format_issue_description(issue)
                    }
                    quality_issues_details.append(issue_detail)
            

            fixes_applied = cleaned.get('_fixes_applied_list', [])
            for fix in fixes_applied:
                fix_detail = {
                    'fix_type': fix.get('type', 'unknown'),
                    'original': fix.get('original', ''),
                    'expanded': fix.get('expanded', ''),
                    'description': f"Expanded '{fix.get('original', '')}' to '{fix.get('expanded', '')}'"
                }
                quality_issues_details.append(fix_detail)
            

            if quality_issues_details:
                cleaned['_quality_issues'] = json.dumps(quality_issues_details, ensure_ascii=False)
            else:
                cleaned['_quality_issues'] = ''
            

            cleaned.pop('_fixes_applied_list', None)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Processed record in {processing_time}s")
            
            self.stats_recorder.increment('processed_records')
            return cleaned
        
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            return None
    
    def _format_issue_description(self, issue: Dict[str, Any]) -> str:

        issue_type = issue.get('type', 'unknown')
        details = issue.get('details', {})
        
        if issue_type == 'abbreviations_detected':
            abbrevs = details.get('abbreviations', [])
            count = details.get('count', 0)
            return f"Found {count} abbreviation(s): {', '.join(abbrevs)}"
        elif issue_type == 'censored_words_detected':
            patterns = details.get('patterns', [])
            count = details.get('count', 0)
            return f"Found {count} censored word pattern(s): {', '.join(patterns[:3])}"
        elif issue_type == 'mixed_characters':
            return "Text contains mixed alphanumeric/special character patterns"
        elif issue_type == 'multiple_short_forms':
            count = details.get('short_word_count', 0) if isinstance(details, dict) else 0
            return f"Found {count} short words (potential abbreviations)"
        elif issue_type == 'very_short':
            return "Comment is very short (less than 10 characters)"
        elif issue_type == 'encoding_issue':
            return "Text contains encoding issues (invalid characters)"
        elif issue_type == 'low_quality_signal':
            confidence = details.get('confidence', 0) if isinstance(details, dict) else issue.get('confidence', 0)
            return f"Low quality signal detected (confidence: {confidence:.2f})"
        elif isinstance(details, list):
            return f"{issue_type}: {', '.join(str(d) for d in details[:3])}"
        elif isinstance(details, dict):
            return f"{issue_type}: {', '.join(f'{k}={v}' for k, v in list(details.items())[:2])}"
        elif details:
            return f"{issue_type}: {details}"
        else:
            return issue_type
    
    def process_all(self, output_filename: str = None) -> str:

        logger.info("="*60)
        logger.info("Starting TRIAGE AI Agent processing...")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Load data from file or fetch from API
        if self.input_file:
            records = self.load_data_from_file()
        else:
            records = self.fetch_data()
        
        if not records:
            logger.warning("No records found to process")
            return None
        
        # Process records
        cleaned_records = []
        for i, record in enumerate(records):
            logger.info(f"Processing record {i+1}/{len(records)}")
            cleaned = self.process_record(record)
            if cleaned:
                cleaned_records.append(cleaned)
        
        # Generate output filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"triage_cleaned_{timestamp}.csv"
        
        output_path = self.output_dir / output_filename
        

        logger.info(f"Writing {len(cleaned_records)} cleaned records to {output_path}")
        if cleaned_records:

            all_fieldnames = set()
            for record in cleaned_records:
                all_fieldnames.update(record.keys())
            fieldnames = sorted(all_fieldnames)
            

            def prepare_value(value):
                if value is None:
                    return ''
                elif isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                else:
                    return str(value)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in cleaned_records:
                    csv_record = {k: prepare_value(v) for k, v in record.items()}
                    # Ensure all fields are present
                    complete_record = {field: csv_record.get(field, '') for field in fieldnames}
                    writer.writerow(complete_record)
        
        # Save statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats_recorder.set('processing_time_seconds', processing_time)
        
        stats_file = self.output_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.stats_recorder.save_to_file(stats_file)
        
        # Save fix tracking
        fixes_file = self.output_dir / f"fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fixes_file, 'w') as f:
            json.dump({
                'fixes': dict(self.fix_tracker.fixes),
                'statistics': self.fix_tracker.get_statistics()
            }, f, indent=2)
        
        # Print statistics
        self.print_statistics()
        
        logger.info(f"TRIAGE AI processing complete. Output: {output_path}")
        logger.info(f"Statistics saved to: {stats_file}")
        logger.info(f"Fixes tracked in: {fixes_file}")
        logger.info(f"Log file: {self.log_file}")
        
        return str(output_path)
    
    def print_statistics(self):

        stats = self.stats_recorder.get_stats()
        fix_stats = self.fix_tracker.get_statistics()
        
        print("\n" + "="*60)
        print("TRIAGE AI Agent Statistics")
        print("="*60)
        print(f"Total records fetched: {stats['total_records']}")
        print(f"Records processed: {stats['processed_records']}")
        print(f"Records with issues: {stats['records_with_issues']}")
        print(f"Records fixed: {stats['records_fixed']}")
        print(f"Records dropped: {stats['records_dropped']}")
        print(f"AI expansions: {stats['ai_expansions']}")
        print(f"Total fixes applied: {fix_stats['total_fixes']}")
        print(f"Processing time: {stats['processing_time_seconds']:.2f}s")
        print("="*60 + "\n")


def main():

    import argparse
    
    parser = argparse.ArgumentParser(description='TRIAGE AI Agent - Data Quality Assessment and Cleaning')
    parser.add_argument('--api-url', type=str, 
                       default=os.getenv('API_URL', 'http://localhost:5001'),
                       help='API endpoint URL (ignored if --input-file is provided)')
    parser.add_argument('--input-file', type=str, default=None,
                       help='Path to local input file (CSV, JSON, or JSONL). If provided, API will not be used.')
    parser.add_argument('--output-dir', type=str, default='data/cleaned',
                       help='Output directory for cleaned data')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of records to process (default: process all)')
    
    args = parser.parse_args()
    
    # Create and run TRIAGE AI agent
    agent = TriageAIAgent(api_url=args.api_url, input_file=args.input_file, output_dir=args.output_dir, limit=args.limit)
    output_path = agent.process_all(output_filename=args.output_file)
    
    if output_path:
        print(f"\nCleaned data saved to: {output_path}")
        print("Ready for downstream processing.")


if __name__ == '__main__':
    main()
