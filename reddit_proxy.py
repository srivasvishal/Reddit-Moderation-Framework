from flask import Flask, request, Response, redirect
import requests
from bs4 import BeautifulSoup
import re
import json
import csv
import traceback
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import logging

# Import triage agent for comment preprocessing
from triage_agent import TriageAIAgent, FixTracker, StatisticsRecorder

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reddit domain configuration
REDDIT_DOMAIN = 'https://old.reddit.com'
PROXY_HOST = "http://localhost:5000"

# Configuration constants
MIN_COMMENT_LENGTH = 3
MAX_COMMENT_DISPLAY_LENGTH = 500
REQUEST_TIMEOUT = 30
CONFIDENCE_THRESHOLD = 80

# Server-side cookie storage for Reddit session persistence
reddit_cookie_jar = {}

# Track last accessed URL
last_accessed_url = None

# Global toxicity predictor instance (loaded once at startup)
toxicity_predictor = None
PREDICTOR_LOADED = False

# Global triage agent for comment preprocessing
triage_agent = None
TRIAGE_LOADED = False
USE_TRIAGE = True  # Set to False to disable triage preprocessing


# =============================================================================
# Helper Functions
# =============================================================================

def get_spoofed_headers(original_headers, target_url):
    """Spoof headers to appear as legitimate Reddit traffic."""
    parsed = urlparse(target_url)
    target_host = parsed.netloc

    # Filter problematic headers
    excluded = ['host', 'connection', 'origin', 'referer']
    headers = {k: v for k, v in original_headers if k.lower() not in excluded}

    # Add spoofed headers
    headers['Host'] = target_host
    headers['Origin'] = f"https://{target_host}"

    # Spoof Referer - convert proxy URLs to Reddit URLs
    original_referer = dict(original_headers).get('Referer', '')
    if original_referer and PROXY_HOST in original_referer:
        headers['Referer'] = original_referer.replace(PROXY_HOST, f"https://{target_host}")
    else:
        headers['Referer'] = f"https://{target_host}/"

    # Ensure proper User-Agent
    if 'User-Agent' not in headers:
        headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

    return headers


def store_reddit_cookies(response_cookies):
    """Store cookies from Reddit response in server-side jar."""
    for cookie in response_cookies:
        reddit_cookie_jar[cookie.name] = cookie.value
    if response_cookies:
        logger.info(f"Stored cookies: {list(reddit_cookie_jar.keys())}")


def get_reddit_cookies():
    """Get stored Reddit cookies."""
    return dict(reddit_cookie_jar)


def merge_cookies(browser_cookies, stored_cookies):
    """Merge browser cookies with stored Reddit cookies. Browser takes precedence."""
    merged = dict(stored_cookies)
    merged.update(browser_cookies)
    return merged


def get_toxicity_predictor():
    """
    Load the toxicity predictor. Returns the predictor or None if loading fails.
    Predictor is loaded once and kept in memory.
    """
    global toxicity_predictor, PREDICTOR_LOADED

    if PREDICTOR_LOADED:
        return toxicity_predictor

    try:
        logger.info("Loading toxicity predictor...")

        from toxicity_predictor import ToxicityPredictor

        model_path = Path(__file__).parent / "models" / "bert_model_full_data"

        if model_path.exists():
            toxicity_predictor = ToxicityPredictor(str(model_path))
            logger.info("Toxicity predictor loaded successfully!")

            adapters = toxicity_predictor.get_available_adapters()
            if adapters:
                logger.info(f"Available subreddit adapters: {list(adapters.keys())}")
        else:
            logger.warning(f"Model not found at {model_path}, predictions disabled")
            toxicity_predictor = None

        PREDICTOR_LOADED = True
        return toxicity_predictor

    except Exception as e:
        logger.error(f"Failed to load toxicity predictor: {e}")
        logger.error(traceback.format_exc())
        PREDICTOR_LOADED = True
        toxicity_predictor = None
        return None


def get_triage_agent():
    """
    load the TriageAIAgent for comment preprocessing.
    Agent is loaded once and kept in memory.
    """
    global triage_agent, TRIAGE_LOADED

    # Skip loading if disabled
    if not USE_TRIAGE:
        logger.info("Triage disabled (USE_TRIAGE=False)")
        TRIAGE_LOADED = True
        return None

    if TRIAGE_LOADED:
        return triage_agent

    try:
        logger.info("=" * 50)
        logger.info("Loading TriageAIAgent...")
        logger.info("=" * 50)

        # Use dashboard_data as output dir for stats
        output_dir = Path(__file__).parent / "dashboard_data"
        triage_agent = TriageAIAgent(output_dir=str(output_dir))

        if triage_agent.ai_expander.api_available:
            logger.info(f"TriageAIAgent loaded (Ollama model: {triage_agent.ai_expander.model_name})")
        else:
            logger.warning("TriageAIAgent loaded but Ollama not available - will skip preprocessing")

        TRIAGE_LOADED = True
        logger.info("=" * 50)
        return triage_agent

    except Exception as e:
        logger.error(f"Failed to load TriageAIAgent: {e}")
        logger.error(traceback.format_exc())
        TRIAGE_LOADED = True  # Mark as loaded to prevent repeated attempts
        triage_agent = None
        return None


def save_triage_progress(current, total, status="processing"):
    """Save triage progress for Streamlit to display."""
    data_dir = Path(__file__).parent / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    progress_file = data_dir / "triage_progress.json"

    with open(progress_file, 'w') as f:
        json.dump({
            'current': current,
            'total': total,
            'status': status,  # 'processing', 'complete', 'idle'
            'timestamp': datetime.now().isoformat()
        }, f)


def save_triage_stats(triage, cleaned_records):
    """Save triage stats and cleaned comments for current page."""
    data_dir = Path(__file__).parent / "dashboard_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Single files, overwritten each page load (not timestamped)
    stats_file = data_dir / "current_page_stats.json"
    fixes_file = data_dir / "current_page_fixes.json"
    cleaned_file = data_dir / "current_page_cleaned.csv"

    # Get stats from agent's StatisticsRecorder
    stats = triage.stats_recorder.get_stats()
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Get fixes from agent's FixTracker
    fix_data = {
        'fixes': dict(triage.fix_tracker.fixes),
        'statistics': triage.fix_tracker.get_statistics()
    }
    with open(fixes_file, 'w') as f:
        json.dump(fix_data, f, indent=2)

    # Save cleaned comments to CSV
    if cleaned_records:
        fieldnames = ['comment_id', 'original_text', 'cleaned_text', 'was_modified', 'fixes_applied']
        with open(cleaned_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in cleaned_records:
                writer.writerow(record)

    logger.info(f"Saved triage: {len(cleaned_records)} comments, {fix_data['statistics'].get('total_fixes', 0)} fixes")


def run_toxicity_predictions(soup, subreddit=None):
    """
    Extract comments from HTML, preprocess with triage, and run toxicity predictions.
    """
    predictions = {}
    predictor = get_toxicity_predictor()
    triage = get_triage_agent()

    if predictor is None:
        logger.warning("Predictor not available, returning empty predictions")
        return predictions

    # Reset triage stats for this page
    if triage:
        triage.fix_tracker = FixTracker()
        triage.stats_recorder = StatisticsRecorder()

    # Track processing start time
    processing_start = datetime.now()

    if subreddit:
        logger.info(f"Using subreddit-specific predictions for r/{subreddit}")
    else:
        logger.info("Using base model predictions (no subreddit specified)")

    # Find all comment elements
    comments = soup.select('.thing[data-type="comment"]')
    logger.info(f"Found {len(comments)} comments to analyze")

    # Set total records in stats
    if triage:
        triage.stats_recorder.set('total_records', len(comments))

    # Track cleaned records for CSV export
    cleaned_records = []

    for comment in comments:
        try:
            # Get comment ID
            comment_id = comment.get('data-fullname')
            if not comment_id:
                continue

            # Get comment text from .md div
            md_div = comment.select_one('.md')
            if not md_div:
                continue

            original_text = md_div.get_text(strip=True)
            if not original_text or len(original_text) < MIN_COMMENT_LENGTH:
                continue

            # TRIAGE PREPROCESSING using full agent
            cleaned_text = original_text
            fixes = []
            if triage and triage.ai_expander.api_available:
                # Log and save progress for each comment
                comment_num = len(cleaned_records) + 1
                logger.info(f"[Triage {comment_num}/{len(comments)}] Processing comment {comment_id[:15]}...")
                save_triage_progress(comment_num, len(comments), "processing")

                # Create a record dict for the triage agent
                record = {'comment_body': original_text, 'comment_id': comment_id}
                cleaned_record = triage.clean_record(record)
                cleaned_text = cleaned_record.get('comment_body', original_text)
                fixes = cleaned_record.get('_fixes_applied_list', [])

                if fixes:
                    logger.info(f"[Triage {comment_num}/{len(comments)}] Applied {len(fixes)} fix(es)")

            # Track for CSV export
            cleaned_records.append({
                'comment_id': comment_id,
                'original_text': original_text[:MAX_COMMENT_DISPLAY_LENGTH],
                'cleaned_text': cleaned_text[:MAX_COMMENT_DISPLAY_LENGTH],
                'was_modified': original_text != cleaned_text,
                'fixes_applied': json.dumps(fixes)
            })

            # Run prediction on CLEANED text (but store original for display)
            prediction, confidence = predictor.predict(cleaned_text, subreddit=subreddit)

            predictions[comment_id] = {
                "prediction": "toxic" if prediction == 1 else "non-toxic",
                "confidence": int(confidence * 100),  # Convert to percentage
                "comment_text": original_text[:MAX_COMMENT_DISPLAY_LENGTH],
                "cleaned_text": cleaned_text[:MAX_COMMENT_DISPLAY_LENGTH],
                "triage_fixes": fixes,
                "subreddit": subreddit or "base"
            }

            # Increment processed count
            if triage:
                triage.stats_recorder.increment('processed_records')

        except Exception as e:
            logger.warning(f"Error predicting comment {comment_id}: {e}")
            continue

    # Save triage stats to single files (overwrite, not timestamped)
    if triage:
        # Calculate processing time
        processing_time = (datetime.now() - processing_start).total_seconds()
        triage.stats_recorder.set('processing_time_seconds', round(processing_time, 2))

        save_triage_stats(triage, cleaned_records)
        save_triage_progress(len(comments), len(comments), "complete")

    logger.info(f"Generated predictions for {len(predictions)} comments")
    return predictions




def inject_highlight_script(soup, subreddit=None):
    """
    Inject CSS and JavaScript into HTML for toxicity overlay functionality.
    Includes: 3-state highlighting (toxic/safe/uncertain), confidence badges, feedback buttons.
    Uses the fine-tuned BERT model for real toxicity predictions.
    """
    # Run real model predictions on comments in this page
    predictions = run_toxicity_predictions(soup, subreddit=subreddit)
    predictions_json = json.dumps(predictions)
    logger.info(f"Injecting {len(predictions)} predictions for r/{subreddit or 'base'}")

    # CSS for overlay styling
    overlay_css = """
    <style>
    /* Toxicity Overlay Styles - Applied to comment text only */
    .toxicity-text-toxic {
        background-color: rgba(220, 53, 69, 0.25) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .toxicity-text-safe {
        background-color: rgba(40, 167, 69, 0.2) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    .toxicity-text-uncertain {
        background-color: rgba(255, 193, 7, 0.3) !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }

    /* Confidence badge */
    .toxicity-badge {
        display: inline-block;
        padding: 2px 8px;
        margin-left: 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        vertical-align: middle;
    }
    .toxicity-badge.toxic {
        background-color: #dc3545;
        color: white;
    }
    .toxicity-badge.safe {
        background-color: #28a745;
        color: white;
    }
    .toxicity-badge.uncertain {
        background-color: #ffc107;
        color: #333;
    }

    /* Feedback buttons */
    .mod-feedback {
        display: inline-block;
        margin-left: 8px;
    }
    .feedback-btn {
        padding: 2px 8px;
        margin: 0 2px;
        border: 1px solid #ccc;
        border-radius: 3px;
        font-size: 10px;
        cursor: pointer;
        background: #fff;
        transition: all 0.15s ease;
    }
    .feedback-btn:hover {
        transform: scale(1.05);
    }
    .feedback-btn.toxic-btn {
        color: #dc3545;
        border-color: #dc3545;
    }
    .feedback-btn.toxic-btn:hover {
        background: #dc3545;
        color: white;
    }
    .feedback-btn.safe-btn {
        color: #28a745;
        border-color: #28a745;
    }
    .feedback-btn.safe-btn:hover {
        background: #28a745;
        color: white;
    }
    .feedback-btn.selected {
        color: white !important;
    }
    .feedback-btn.toxic-btn.selected {
        background: #dc3545;
    }
    .feedback-btn.safe-btn.selected {
        background: #28a745;
    }
    .feedback-submitted {
        font-size: 10px;
        color: #666;
        margin-left: 4px;
    }
    </style>
    """

    # JavaScript for overlay functionality
    overlay_script = f"""
    <script>
    // Toxicity predictions data
    window.TOXICITY_PREDICTIONS = {predictions_json};

    // Reddit Moderation Tool - Toxicity Overlay
    window.redditModTools = {{
        appliedComments: new Set(),

        applyOverlays: function() {{
            const comments = document.querySelectorAll('.thing[data-type="comment"]');
            let applied = 0;

            comments.forEach(comment => {{
                const fullname = comment.dataset.fullname;
                if (!fullname || this.appliedComments.has(fullname)) return;

                const prediction = window.TOXICITY_PREDICTIONS[fullname];
                if (!prediction) return;

                this.applyOverlayToComment(comment, prediction);
                this.appliedComments.add(fullname);
                applied++;
            }});

            console.log('Applied overlays to ' + applied + ' comments');
            return applied;
        }},

        applyOverlayToComment: function(commentEl, prediction) {{
            const confidence = prediction.confidence;
            const pred = prediction.prediction;

            // Determine state based on confidence threshold
            let state;
            if (confidence < {CONFIDENCE_THRESHOLD}) {{
                state = 'uncertain';
            }} else if (pred === 'toxic') {{
                state = 'toxic';
            }} else {{
                state = 'safe';
            }}

            // Apply CSS class to comment text (md div) only, not the whole comment
            const mdDiv = commentEl.querySelector('.md');
            if (mdDiv) {{
                mdDiv.classList.add('toxicity-text-' + state);
            }}

            // Find tagline and add badge
            const tagline = commentEl.querySelector('.tagline');
            if (tagline && !tagline.querySelector('.toxicity-badge')) {{
                const badge = document.createElement('span');
                badge.className = 'toxicity-badge ' + state;
                badge.textContent = confidence + '% confident';
                badge.title = 'Prediction: ' + pred;
                tagline.appendChild(badge);
            }}

            // Find buttons list and add feedback buttons
            const buttons = commentEl.querySelector('.flat-list.buttons');
            if (buttons && !buttons.querySelector('.mod-feedback')) {{
                const feedbackLi = document.createElement('li');
                feedbackLi.className = 'mod-feedback';
                feedbackLi.innerHTML = `
                    <button class="feedback-btn toxic-btn" onclick="redditModTools.submitFeedback('${{commentEl.dataset.fullname}}', 'toxic', this)">Toxic</button>
                    <button class="feedback-btn safe-btn" onclick="redditModTools.submitFeedback('${{commentEl.dataset.fullname}}', 'non-toxic', this)">Safe</button>
                `;
                buttons.appendChild(feedbackLi);
            }}
        }},

        submitFeedback: function(commentId, humanLabel, btnElement) {{
            const prediction = window.TOXICITY_PREDICTIONS[commentId];

            fetch('/api/feedback', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    comment_id: commentId,
                    comment_text: prediction ? prediction.comment_text : '',
                    original_prediction: prediction ? prediction.prediction : 'unknown',
                    original_confidence: prediction ? prediction.confidence : 0,
                    human_label: humanLabel,
                    thread_url: window.location.href
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    // Mark button as selected
                    const container = btnElement.parentElement;
                    container.querySelectorAll('.feedback-btn').forEach(btn => btn.classList.remove('selected'));
                    btnElement.classList.add('selected');

                    // Add confirmation text
                    if (!container.querySelector('.feedback-submitted')) {{
                        const confirm = document.createElement('span');
                        confirm.className = 'feedback-submitted';
                        confirm.textContent = 'Saved!';
                        container.appendChild(confirm);
                    }}
                    console.log('Feedback submitted for ' + commentId);
                }}
            }})
            .catch(err => console.error('Feedback error:', err));
        }},

        clearHighlights: function() {{
            document.querySelectorAll('.toxicity-text-toxic, .toxicity-text-safe, .toxicity-text-uncertain').forEach(el => {{
                el.classList.remove('toxicity-text-toxic', 'toxicity-text-safe', 'toxicity-text-uncertain');
            }});
            document.querySelectorAll('.toxicity-badge').forEach(el => el.remove());
            document.querySelectorAll('.mod-feedback').forEach(el => el.remove());
            this.appliedComments.clear();
            console.log('Highlights cleared');
        }},

        highlightComments: function() {{
            // Alias for backward compatibility
            return this.applyOverlays();
        }}
    }};

    // Auto-apply overlays when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{
            window.redditModTools.applyOverlays();
        }});
    }} else {{
        window.redditModTools.applyOverlays();
    }}

    // Listen for messages from parent window (Streamlit dashboard)
    window.addEventListener('message', function(event) {{
        if (event.data === 'highlightComments' || event.data === 'applyOverlays') {{
            window.redditModTools.applyOverlays();
        }} else if (event.data === 'clearHighlights') {{
            window.redditModTools.clearHighlights();
        }}
    }});
    </script>
    """

    # Inject CSS into head
    if soup.head:
        css_tag = BeautifulSoup(overlay_css, 'html.parser')
        soup.head.append(css_tag)

    # Inject JavaScript at end of body for better performance
    if soup.body:
        script_tag = BeautifulSoup(overlay_script, 'html.parser')
        soup.body.append(script_tag)
    elif soup.head:
        script_tag = BeautifulSoup(overlay_script, 'html.parser')
        soup.head.append(script_tag)


def extract_subreddit(path):
    """
    Extract subreddit name from a Reddit URL path.
    """
    # Handle paths with or without leading slash (Flask path param has no leading /)
    match = re.search(r'(?:^|/)r/([^/]+)', path)
    if match:
        return match.group(1).lower()
    return None


def rewrite_urls(content, content_type, subreddit=None):
    """
    Rewrite Reddit URLs to point to proxy server.
    """
    if not content:
        return content

    try:
        # HTML content - use BeautifulSoup for accurate parsing
        if 'text/html' in content_type:
            soup = BeautifulSoup(content, 'html.parser')

            # Rewrite href attributes
            for tag in soup.find_all(href=True):
                original = tag['href']
                tag['href'] = rewrite_url(original)

            # Rewrite src attributes
            for tag in soup.find_all(src=True):
                original = tag['src']
                tag['src'] = rewrite_url(original)

            # Rewrite action attributes (forms)
            for tag in soup.find_all(action=True):
                original = tag['action']
                tag['action'] = rewrite_url(original)

            # Rewrite inline styles with url()
            for tag in soup.find_all(style=True):
                tag['style'] = rewrite_css_urls(tag['style'])

            # Rewrite style tags
            for style_tag in soup.find_all('style'):
                if style_tag.string:
                    style_tag.string = rewrite_css_urls(style_tag.string)

            # Rewrite script tags that might contain URLs
            for script_tag in soup.find_all('script'):
                if script_tag.string:
                    script_tag.string = rewrite_js_urls(script_tag.string)

            # Inject comment highlighting JavaScript (with subreddit-specific adapter)
            inject_highlight_script(soup, subreddit=subreddit)

            return str(soup)

        # CSS content
        elif 'text/css' in content_type:
            return rewrite_css_urls(content)

        # JavaScript content
        elif 'javascript' in content_type or 'application/json' in content_type:
            return rewrite_js_urls(content)

        # Other content types - return as-is
        else:
            return content

    except Exception as e:
        logger.error(f"Error rewriting URLs: {e}")
        return content


def _get_all_reddit_domains():
    """Generate all Reddit domain variants (https, http, protocol-relative)."""
    domain = REDDIT_DOMAIN.replace('https://', '')
    return [
        f'https://{domain}',
        f'http://{domain}',
        f'//{domain}',
    ]


def rewrite_url(url):
    """Convert any Reddit URL to proxy URL."""
    if not url:
        return url

    if url.startswith(PROXY_HOST):
        return url

    for domain in _get_all_reddit_domains():
        if url.startswith(domain):
            return url.replace(domain, PROXY_HOST)

    # Relative or external URLs - keep as-is
    if url.startswith('/') or not url.startswith('http'):
        return url

    return url


def rewrite_css_urls(css_content):
    """
    Rewrite URLs in CSS url() declarations.
    """
    if not css_content:
        return css_content

    # Match url(...) patterns
    def replace_url(match):
        original_url = match.group(1).strip('\'"')
        new_url = rewrite_url(original_url)
        return f'url({new_url})'

    return re.sub(r'url\(["\']?([^"\')]+)["\']?\)', replace_url, css_content)


def rewrite_js_urls(js_content):
    """Rewrite all Reddit URLs in JavaScript code."""
    if not js_content:
        return js_content

    # Generate replacements for quoted URLs in JS
    for domain in _get_all_reddit_domains():
        js_content = js_content.replace(f'"{domain}', f'"{PROXY_HOST}')
        js_content = js_content.replace(f"'{domain}", f"'{PROXY_HOST}")

    return js_content


# =============================================================================
# API Endpoints for Toxicity Overlay
# =============================================================================



@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Receive human feedback on toxicity predictions and store in CSV.
    """
    try:
        data = request.get_json()

        if not data or 'comment_id' not in data:
            return Response(
                json.dumps({'success': False, 'error': 'Missing comment_id'}),
                status=400,
                mimetype='application/json'
            )

        # Extract subreddit from thread_url
        thread_url = data.get('thread_url', '')
        subreddit = ''
        if '/r/' in thread_url:
            parts = thread_url.split('/r/')
            if len(parts) > 1:
                subreddit = parts[1].split('/')[0]

        # Build feedback entry
        feedback_entry = {
            'feedback_id': f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data['comment_id'][-6:]}",
            'comment_id': data['comment_id'],
            'comment_text': data.get('comment_text', ''),
            'original_prediction': data.get('original_prediction', ''),
            'original_confidence': data.get('original_confidence', 0),
            'human_label': data.get('human_label', ''),
            'feedback_timestamp': datetime.now().isoformat(),
            'thread_url': thread_url,
            'subreddit': subreddit
        }

        # Save to CSV
        csv_path = Path(__file__).parent / "dashboard_data" / "human_feedback.csv"
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_entry)

        logger.info(f"Feedback saved: {feedback_entry['feedback_id']} - {data['human_label']}")

        return Response(
            json.dumps({'success': True, 'feedback_id': feedback_entry['feedback_id']}),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return Response(
            json.dumps({'success': False, 'error': str(e)}),
            status=500,
            mimetype='application/json'
        )


# =============================================================================
# Main Proxy Route
# =============================================================================

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy(path):
    """Main proxy endpoint - forwards all requests to Reddit."""
    target_url = f"{REDDIT_DOMAIN}/{path}"

    # Add query parameters
    if request.query_string:
        target_url += f"?{request.query_string.decode('utf-8')}"

    logger.info(f"{request.method} {target_url}")

    # Use spoofed headers to appear as legitimate Reddit traffic
    headers = get_spoofed_headers(request.headers, target_url)

    # Merge browser cookies with stored Reddit session cookies
    browser_cookies = dict(request.cookies)
    stored_cookies = get_reddit_cookies()
    all_cookies = merge_cookies(browser_cookies, stored_cookies)

    try:
        # Forward the request to Reddit with merged cookies
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=request.get_data(),
            cookies=all_cookies,
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT
        )

        # Store cookies from Reddit response
        store_reddit_cookies(resp.cookies)

        # Handle redirects with cookie preservation
        if resp.status_code in [301, 302, 303, 307, 308]:
            location = resp.headers.get('Location', '')
            if location:
                # Rewrite the redirect location to proxy
                new_location = rewrite_url(location)

                # Log for debugging
                logger.info(f"Redirect: {location} -> {new_location}")

                # Check if this is a post-login redirect (has reddit_session cookie)
                if resp.cookies.get('reddit_session'):
                    logger.info("Post-login redirect detected - session cookie received!")

                redirect_response = redirect(new_location, code=resp.status_code)

                # Set cookies on redirect response
                for cookie in resp.cookies:
                    redirect_response.set_cookie(
                        key=cookie.name,
                        value=cookie.value,
                        path=cookie.path or '/',
                        samesite='Lax'
                    )

                return redirect_response

        # Build response headers
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection',
                          'x-frame-options', 'content-security-policy', 'x-content-security-policy']
        response_headers = {
            key: value for key, value in resp.headers.items()
            if key.lower() not in excluded_headers
        }

        # Add CORS headers to allow embedding
        response_headers['Access-Control-Allow-Origin'] = '*'
        response_headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response_headers['Access-Control-Allow-Headers'] = '*'

        # Add cache-control headers to prevent browser caching
        response_headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response_headers['Pragma'] = 'no-cache'
        response_headers['Expires'] = '0'

        # Get content type
        content_type = resp.headers.get('Content-Type', '')

        # Rewrite content if applicable
        if resp.content and any(ct in content_type for ct in ['text/html', 'text/css', 'javascript', 'application/json']):
            try:
                content = resp.content.decode('utf-8', errors='ignore')

                # Extract subreddit from path for adapter selection
                subreddit = extract_subreddit(path)
                if subreddit:
                    logger.info(f"Detected subreddit: r/{subreddit}")

                rewritten_content = rewrite_urls(content, content_type, subreddit=subreddit)
                response_content = rewritten_content.encode('utf-8')

                # Track last accessed URL
                if 'text/html' in content_type and request.method == 'GET':
                    global last_accessed_url
                    last_accessed_url = target_url

            except Exception as e:
                logger.error(f"Error rewriting content: {e}")
                response_content = resp.content
        else:
            response_content = resp.content

        # Create response
        response = Response(
            response_content,
            status=resp.status_code,
            headers=response_headers
        )

        # Forward cookies to browser (cookies already stored server-side above)
        for cookie in resp.cookies:
            response.set_cookie(
                key=cookie.name,
                value=cookie.value,
                path=cookie.path or '/',
                secure=False,  # Use HTTP for localhost
                httponly=cookie.has_nonstandard_attr('HttpOnly'),
                samesite='Lax'  # Allow cookies with navigations
            )

        return response

    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy error: {e}")
        return Response(f"Proxy Error: {str(e)}", status=502)


@app.route('/proxy/adapters', methods=['GET'])
def list_adapters():
    """List available subreddit LoRA adapters."""
    global last_accessed_url
    predictor = get_toxicity_predictor()

    if predictor is None:
        return Response(
            json.dumps({'error': 'Predictor not loaded', 'adapters': []}),
            status=200,
            mimetype='application/json'
        )

    adapters = predictor.get_available_adapters()

    # Extract subreddit from last accessed URL for accurate display
    current_subreddit = None
    if last_accessed_url:
        current_subreddit = extract_subreddit(last_accessed_url)

    # Check if there's an adapter for the current subreddit
    adapter_active = current_subreddit and current_subreddit in adapters

    result = {
        'adapters': list(adapters.keys()),
        'count': len(adapters),
        'current_adapter': current_subreddit if adapter_active else None,
        'current_subreddit': current_subreddit,
        'loaded_adapters': list(predictor.adapted_models.keys())
    }
    return Response(json.dumps(result, indent=2), status=200, mimetype='application/json')


@app.route('/proxy/current-url', methods=['GET'])
def get_current_url():
    """
    Returns the last accessed Reddit URL.
    """
    global last_accessed_url
    if last_accessed_url:
        return Response(last_accessed_url, status=200, mimetype='text/plain')
    else:
        return Response("No URL accessed yet", status=404)


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Reddit Proxy Server Starting")
    logger.info(f"Proxying: {REDDIT_DOMAIN}")
    logger.info(f"Local URL: {PROXY_HOST}")
    logger.info("=" * 60)

    # Preload predictor at startup
    logger.info("Preloading toxicity predictor...")
    predictor = get_toxicity_predictor()
    if predictor:
        logger.info("Predictor ready!")
    else:
        logger.warning("Predictor failed to load - predictions disabled")

    # Preload triage agent at startup
    if USE_TRIAGE:
        logger.info("Preloading triage agent...")
        triage = get_triage_agent()
        if triage and triage.ai_expander.api_available:
            logger.info(f"Triage ready! (Ollama model: {triage.ai_expander.model_name})")
        elif triage:
            logger.warning("Triage loaded but Ollama not available")
        else:
            logger.warning("Triage failed to load")

    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
