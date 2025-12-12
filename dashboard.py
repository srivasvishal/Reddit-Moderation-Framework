# Google Chrome extension: Ignore X-Frame headers 
# Load Streamlit on Google Chrome make sure to have extension enabled for the iframe to work properly


import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import plotly.graph_objects as go
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import time

# Page configuration
st.set_page_config(
    page_title="Reddit Moderation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation persistence
if 'current_reddit_url' not in st.session_state:
    st.session_state.current_reddit_url = None
if 'nav_action' not in st.session_state:
    st.session_state.nav_action = None
if 'nav_timestamp' not in st.session_state:
    st.session_state.nav_timestamp = 0

# CSS styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f4788;
        --secondary-color: #4a90e2;
        --accent-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --background-light: #f8f9fa;
        --text-dark: #2c3e50;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1f4788 0%, #4a90e2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }

    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Section styling */
    .section-header {
        color: var(--text-dark);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary-color);
    }

    /* Info box styling */
    .info-box {
        background: var(--background-light);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--secondary-color);
        margin: 1rem 0;
    }

    /* File info styling */
    .file-info {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        font-family: monospace;
        font-size: 0.9rem;
        color: #555;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    /* Streamlit metric override */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--primary-color);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


def load_json_file(file_path: Path) -> Optional[Dict]:
    """
    Load a JSON file with robust encoding handling.
    """
    # Try multiple encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                data = json.load(f)
                return data
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    # If all encodings fail, try reading as binary and decoding with error handling
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # Try to decode with utf-8, replacing errors
            text = content.decode('utf-8', errors='replace')
            data = json.loads(text)
            return data
    except Exception as e:
        st.error(f"Failed to load {file_path.name}: {str(e)}")
        return None


def load_latest_files(data_dir: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load current page stats and fixes from live proxy."""
    try:
        current_stats = data_dir / "current_page_stats.json"
        current_fixes = data_dir / "current_page_fixes.json"

        if not current_stats.exists() or not current_fixes.exists():
            return None, None

        stats_data = load_json_file(current_stats)
        fixes_data = load_json_file(current_fixes)

        if stats_data is None or fixes_data is None:
            return None, None

        return stats_data, fixes_data

    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return None, None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def render_header():
    """Render the main dashboard header."""
    st.markdown("""
        <div class="main-header">
            <h1>Reddit Moderation System Dashboard</h1>
            <p>Real-time Toxicity Detection with AI-Powered Text Preprocessing</p>
        </div>
    """, unsafe_allow_html=True)


def render_navigation_bar():
    """Render navigation controls above the iframe."""

    if st.button("Refresh", key="nav_refresh"):
        # Get current URL for refresh (don't go to homepage)
        current = fetch_current_url()
        if current:
            st.session_state.current_reddit_url = current
        st.session_state.nav_action = 'refresh'
        st.session_state.nav_timestamp = time.time()
        st.rerun()


def render_reddit_overlay():
    """Render the Reddit Overlay section with live old.reddit.com embed via proxy."""
    st.markdown('<p class="section-header">Reddit Overlay</p>', unsafe_allow_html=True)

    # Show triage progress (updates every second during processing)
    render_triage_progress()

    # Show adapter status above iframe
    render_adapter_status()

    # Sync current URL to session state (for persistence)
    sync_current_url()

    # On initial load, fetch current URL from proxy if session state is empty
    if not st.session_state.current_reddit_url:
        current = fetch_current_url()
        if current:
            st.session_state.current_reddit_url = current

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Render navigation bar
    render_navigation_bar()

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    # Determine which URL to load in iframe
    iframe_url = "http://localhost:5000"

    # Check if we have a navigation action (refresh)
    nav_action = st.session_state.get('nav_action')
    nav_timestamp = st.session_state.get('nav_timestamp', 0)

    if nav_action == 'refresh' and st.session_state.current_reddit_url:
        reddit_url = st.session_state.current_reddit_url
        if reddit_url.startswith('https://old.reddit.com'):
            path = reddit_url.replace('https://old.reddit.com', '')
            iframe_url = f"http://localhost:5000{path}"
        iframe_url = f"{iframe_url}{'&' if '?' in iframe_url else '?'}_refresh={int(time.time())}"
        st.session_state.nav_action = None
    elif nav_action == 'refresh':
        iframe_url = f"http://localhost:5000?_refresh={int(time.time())}"
        st.session_state.nav_action = None
    elif st.session_state.current_reddit_url:
        reddit_url = st.session_state.current_reddit_url
        if reddit_url.startswith('https://old.reddit.com'):
            path = reddit_url.replace('https://old.reddit.com', '')
            iframe_url = f"http://localhost:5000{path}"
        iframe_url = f"{iframe_url}{'&' if '?' in iframe_url else '?'}_t={int(nav_timestamp)}"

    # Embed the iframe using components.html for better control
    iframe_html = f'''
    <iframe
        id="reddit-iframe"
        src="{iframe_url}"
        style="width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 8px;"
        scrolling="yes"
    ></iframe>
    '''
    components.html(iframe_html, height=820, scrolling=False)


def render_overview_metrics(stats_data: Dict):
    """Render the overview metrics section with key statistics."""
    st.markdown('<p class="section-header">Overview Metrics</p>', unsafe_allow_html=True)

    # Calculate derived metrics
    processing_time = stats_data.get('processing_time_seconds', 0)
    total_records = stats_data.get('total_records', 0)
    records_per_sec = total_records / processing_time if processing_time > 0 else 0

    processed = stats_data.get('processed_records', 0)
    with_issues = stats_data.get('records_with_issues', 0)
    clean_records = processed - with_issues
    success_rate = (clean_records / processed * 100) if processed > 0 else 0

    # Row 1: Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{total_records:,}",
            delta=f"{processed} processed"
        )

    with col2:
        st.metric(
            label="Processing Time",
            value=format_duration(processing_time),
            delta=f"{records_per_sec:.1f} rec/sec"
        )

    with col3:
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1f}%",
            delta=f"{clean_records} clean"
        )

    with col4:
        st.metric(
            label="Total Fixes",
            value=f"{stats_data.get('records_fixed', 0):,}",
            delta=f"{with_issues} with issues"
        )

    # Row 2: Fix type metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="AI Expansions",
            value=f"{stats_data.get('ai_expansions', 0):,}",
            help="Text expansions performed using AI"
        )

    with col2:
        st.metric(
            label="Rule-Based Expansions",
            value=f"{stats_data.get('rule_based_expansions', 0):,}",
            help="Text expansions using predefined rules"
        )

    with col3:
        st.metric(
            label="Records with Issues",
            value=f"{with_issues:,}",
            delta=f"{(with_issues/total_records*100):.1f}%" if total_records > 0 else "0%",
            help="Records that required fixes"
        )

    with col4:
        st.metric(
            label="Records Dropped",
            value=f"{stats_data.get('records_dropped', 0):,}",
            delta="Failures" if stats_data.get('records_dropped', 0) > 0 else "None",
            delta_color="inverse"
        )


def render_fix_type_visualizations(fixes_data: Dict):
    """Render visualizations for fix type distribution."""
    st.markdown('<p class="section-header">Fix Type Analysis</p>', unsafe_allow_html=True)

    # Extract fix type statistics
    fixes_by_type = fixes_data.get('statistics', {}).get('fixes_by_type', {})

    if not fixes_by_type:
        st.warning("No fix type data available.")
        return

    # Prepare data for visualization
    fix_types = list(fixes_by_type.keys())
    fix_counts = list(fixes_by_type.values())

    # Format labels nicely
    formatted_labels = []
    for ft in fix_types:
        formatted = ft.replace('_', ' ').title()
        formatted_labels.append(formatted)

    # Create two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart for distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=formatted_labels,
            values=fix_counts,
            hole=0.4,
            marker=dict(
                colors=['#1f4788', '#4a90e2', '#2ecc71', '#f39c12', '#e74c3c'],
                line=dict(color='white', width=2)
            ),
            textinfo='percent',
            textposition='outside',
            textfont=dict(size=11, color='#2c3e50'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig_pie.update_layout(
            title={
                'text': 'Fix Type Distribution',
                'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            height=400,
            margin=dict(t=60, b=80, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart for counts
        fig_bar = go.Figure(data=[go.Bar(
            x=formatted_labels,
            y=fix_counts,
            marker=dict(
                color=fix_counts,
                colorscale='Blues',
                showscale=False,
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=fix_counts,
            textposition='outside',
            textfont=dict(size=14, color='#2c3e50'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )])

        fig_bar.update_layout(
            title={
                'text': 'Fix Counts by Type',
                'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Fix Type',
                showgrid=False,
                tickangle=-45
            ),
            yaxis=dict(
                title='Count',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Summary statistics table
    st.markdown("#### Fix Type Summary")

    total_fixes = sum(fix_counts)
    summary_data = []

    for label, count in zip(formatted_labels, fix_counts):
        percentage = (count / total_fixes * 100) if total_fixes > 0 else 0
        summary_data.append({
            'Fix Type': label,
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })

    df = pd.DataFrame(summary_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fix Type": st.column_config.TextColumn("Fix Type", width="medium"),
            "Count": st.column_config.NumberColumn("Count", width="small"),
            "Percentage": st.column_config.TextColumn("Percentage", width="small")
        }
    )


def render_processing_timeline(stats_data: Dict):
    """Render processing timeline information."""
    st.markdown('<p class="section-header">Processing Timeline</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    start_time = stats_data.get('start_time', 'N/A')
    end_time = stats_data.get('end_time', 'N/A')
    duration = stats_data.get('processing_time_seconds', 0)

    with col1:
        st.markdown(f"""
            <div class="info-box">
                <strong>Start Time</strong><br>
                {format_timestamp(start_time)}
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="info-box">
                <strong>End Time</strong><br>
                {format_timestamp(end_time)}
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="info-box">
                <strong>Total Duration</strong><br>
                {format_duration(duration)}
            </div>
        """, unsafe_allow_html=True)


def fetch_adapter_info() -> Optional[Dict]:
    """
    Fetch available LoRA adapters from the proxy server.
    """
    try:
        response = requests.get("http://localhost:5000/proxy/adapters", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def fetch_current_url() -> Optional[str]:
    """Fetch the current URL from proxy server."""
    try:
        response = requests.get("http://localhost:5000/proxy/current-url", timeout=2)
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException:
        pass
    return None


@st.fragment(run_every=4)
def render_adapter_status():
    """Render adapter status above the Reddit iframe. Auto-refreshes every 2 seconds."""
    adapter_info = fetch_adapter_info()

    if adapter_info is None:
        st.warning("Proxy server not running. Start with: `python agents/reddit_proxy.py`")
        return

    adapters = adapter_info.get('adapters', [])
    current = adapter_info.get('current_adapter')

    # Create columns for status display
    col1, col2 = st.columns([2, 3])

    with col1:
        if current:
            st.markdown(f"""
                <div style="background: #d4edda; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #28a745;">
                    <strong>Current Subreddit:</strong> r/{current}
                    <br><small style="color: #155724;">Using subreddit-specific adapter</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: #e2e3e5; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #6c757d;">
                    <strong>Current Subreddit:</strong> None / Base Model
                    <br><small style="color: #383d41;">Using base toxicity model</small>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        if adapters:
            adapter_list = ", ".join([f"r/{a}" for a in sorted(adapters)])
            st.markdown(f"""
                <div style="background: #d1ecf1; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #17a2b8;">
                    <strong>Available Adapters ({len(adapters)}):</strong> {adapter_list}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: #fff3cd; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <strong>No adapters trained</strong>
                    <br><small>Run: python agents/train_subreddit_adapters.py</small>
                </div>
            """, unsafe_allow_html=True)


@st.fragment(run_every=3)
def sync_current_url():
    """Periodically sync the current URL from proxy to session state."""
    current_url = fetch_current_url()
    if current_url and current_url != st.session_state.get('current_reddit_url'):
        st.session_state.current_reddit_url = current_url


@st.fragment(run_every=1)
def render_triage_progress():
    """Render triage preprocessing progress. Auto-refreshes every second."""
    progress_file = Path(__file__).parent / "dashboard_data" / "triage_progress.json"

    if not progress_file.exists():
        return

    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)

        status = progress.get('status', 'idle')
        current = progress.get('current', 0)
        total = progress.get('total', 0)

        if status == 'processing' and total > 0:
            pct = int((current / total) * 100)
            st.markdown(f"""
                <div style="background: #fff3cd; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 0.5rem;">
                    <strong>Triage Processing:</strong> {current}/{total} comments ({pct}%)
                    <div style="background: #e0e0e0; border-radius: 4px; height: 8px; margin-top: 4px;">
                        <div style="background: #ffc107; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        elif status == 'complete' and total > 0:
            # Check if recently completed (within 5 seconds)
            timestamp = progress.get('timestamp', '')
            try:
                completed_time = datetime.fromisoformat(timestamp)
                if (datetime.now() - completed_time).total_seconds() < 5:
                    st.markdown(f"""
                        <div style="background: #d4edda; padding: 0.5rem 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 0.5rem;">
                            <strong>Triage Complete:</strong> {total} comments preprocessed
                        </div>
                    """, unsafe_allow_html=True)
            except:
                pass

    except Exception:
        pass  # Silently fail if file is being written


@st.fragment(run_every=10)
def render_statistics_dashboard():
    """Render statistics dashboard with auto-refresh. Only re-renders when data changes."""
    data_dir = Path(__file__).parent / "dashboard_data"

    if not data_dir.exists():
        st.warning("Waiting for data... Stats will appear after browsing Reddit.")
        return

    # Check file modification times to detect changes
    stats_file = data_dir / "current_page_stats.json"
    fixes_file = data_dir / "current_page_fixes.json"

    # Get current modification times
    current_mtime = None
    if stats_file.exists() and fixes_file.exists():
        current_mtime = (stats_file.stat().st_mtime, fixes_file.stat().st_mtime)

    # Initialize session state for tracking
    if 'stats_mtime' not in st.session_state:
        st.session_state.stats_mtime = None
        st.session_state.cached_stats = None
        st.session_state.cached_fixes = None

    # Check if data changed or first load
    data_changed = current_mtime != st.session_state.stats_mtime

    if data_changed:
        # Load fresh data
        stats_data, fixes_data = load_latest_files(data_dir)
        if stats_data is not None and fixes_data is not None:
            st.session_state.cached_stats = stats_data
            st.session_state.cached_fixes = fixes_data
            st.session_state.stats_mtime = current_mtime

    # Use cached data for rendering
    stats_data = st.session_state.cached_stats
    fixes_data = st.session_state.cached_fixes

    if stats_data is None or fixes_data is None:
        st.info("No triage data yet. Browse Reddit pages to generate statistics.")
        return

    # Render all statistics sections
    render_overview_metrics(stats_data)
    render_fix_type_visualizations(fixes_data)
    render_processing_timeline(stats_data)


def main():
    """Main dashboard application."""

    # Render header
    render_header()

    # Define data directory
    data_dir = Path(__file__).parent / "dashboard_data"

    # Check if directory exists
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Render Reddit Overlay section
    render_reddit_overlay()

    # Render auto-refreshing statistics dashboard
    render_statistics_dashboard()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>"
        "Reddit Moderation System Dashboard | Powered by Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
