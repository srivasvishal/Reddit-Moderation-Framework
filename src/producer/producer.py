from time import sleep
from json import dumps, loads
import json
from flask import Flask, Response, request, jsonify
import praw
import sys
from pathlib import Path
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


try:
    from src.monitoring.kafka_monitor import KafkaMonitoringProducer
    MONITORING_AVAILABLE = True
except ImportError:
    from kafka import KafkaProducer
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring not available, using standard producer")

app = Flask(__name__)


if MONITORING_AVAILABLE:
    monitoring_producer = KafkaMonitoringProducer(
        bootstrap_servers=Config.get_kafka_servers(),
        topic=Config.KAFKA_TOPIC
    )
else:
    from kafka import KafkaProducer
    standard_producer = KafkaProducer(
        bootstrap_servers=Config.get_kafka_servers(),
        value_serializer=lambda x: dumps(x).encode('utf-8')
    )

def check_reddit_rate_limits(r, subreddit_name, requested_limit, is_comment_mode=False):
    """
    Check Reddit API rate limits and determine maximum possible retrieval limit.
    Prints header information and rate limit details.
    """
    
    REDDIT_MAX_LISTING_LIMIT = 1000
    

    if is_comment_mode:
        
        COMMENT_STREAMING_WARNING_THRESHOLD = 50000  
    
    print("\n" + "="*70)
    print("REDDIT API RATE LIMIT CHECK")
    print("="*70)
    
    rate_limit_info = {
        'requested_limit': requested_limit,
        'max_listing_limit': REDDIT_MAX_LISTING_LIMIT,
        'rate_limit_used': None,
        'rate_limit_remaining': None,
        'rate_limit_reset': None,
        'adjusted_limit': None
    }
    
    try:
        
        subreddit = r.subreddit(subreddit_name)
        
        
        remaining_requests = None
        reset_timestamp = None
        used_requests = None
        
        try:
            
            _ = list(subreddit.hot(limit=1))
            
            
            if hasattr(r, 'auth') and hasattr(r.auth, 'limits'):
                limits = r.auth.limits
                if limits:
                    remaining_requests = limits.get('remaining')
                    used_requests = limits.get('used')
                    reset_timestamp = limits.get('reset_timestamp')
                    
                    rate_limit_info['rate_limit_remaining'] = remaining_requests
                    rate_limit_info['rate_limit_used'] = used_requests
                    rate_limit_info['rate_limit_reset'] = reset_timestamp
                    
                    
                    print(f"\n REDDIT API RATE LIMIT STATUS:")
                    print(f"   Requests Used: {used_requests if used_requests is not None else 'N/A'}")
                    print(f"   Requests Remaining (Pending): {remaining_requests if remaining_requests is not None else 'N/A'}")
                    
                    if reset_timestamp:
                        reset_time = datetime.fromtimestamp(reset_timestamp)
                        reset_time_str = reset_time.strftime('%Y-%m-%d %H:%M:%S')
                        time_until_reset = reset_time - datetime.now()
                        minutes_until_reset = int(time_until_reset.total_seconds() / 60)
                        
                        print(f"   Rate Limit Resets At: {reset_time_str}")
                        print(f"   Time Until Reset: {minutes_until_reset} minutes")
                        
                        
                        if remaining_requests is not None:
                            if remaining_requests <= 0:
                                print(f"\n  CRITICAL: No requests remaining! Rate limit has been exhausted.")
                                print(f"   Please wait until {reset_time_str} for the limit to refresh.")
                            elif remaining_requests < 10:
                                print(f"\n  WARNING: Very few requests remaining ({remaining_requests}).")
                                print(f"   Limit will refresh at {reset_time_str} ({minutes_until_reset} minutes).")
                            else:
                                print(f"\n Sufficient requests available ({remaining_requests} remaining).")
                    else:
                        print(f"   Rate Limit Reset Time: Not available")
        except Exception as e:
            print(f"  Warning: Could not access rate limit info via PRAW: {e}")
            print(f"   Proceeding with requested limit, but rate limits will be monitored during streaming.")
        
        
        print(f"\n REQUEST VALIDATION:")
        print(f"   Requested Limit: {requested_limit}")
        
        if is_comment_mode:
            print(f"\n Comment Streaming Mode Detected")
            print(f"   Reddit API Maximum Submissions Limit: {REDDIT_MAX_LISTING_LIMIT}")
            print(f"   Requested Comment Limit: {requested_limit}")
            
            
            estimated_requests_needed = min(requested_limit // 100, REDDIT_MAX_LISTING_LIMIT) if requested_limit > 100 else 10
            
            if remaining_requests is not None:
                if remaining_requests < estimated_requests_needed:
                    print(f"\n  WARNING: Remaining requests ({remaining_requests}) may be insufficient")
                    print(f"   Estimated requests needed: ~{estimated_requests_needed}")
                    print(f"   The system will monitor rate limits during streaming and may stop early if limits are reached.")
                else:
                    print(f"\n Sufficient requests available for comment streaming")
                    print(f"   Remaining: {remaining_requests}, Estimated needed: ~{estimated_requests_needed}")
            
            
            if requested_limit > COMMENT_STREAMING_WARNING_THRESHOLD:
                print(f"\n  WARNING: Requested comment limit ({requested_limit}) is very high.")
                print(f"   This may take a long time and could hit rate limits.")
                print(f"   The system will monitor rate limits during streaming and adjust as needed.")
                print(f"   Maximum submissions that can be processed: {REDDIT_MAX_LISTING_LIMIT}")
                adjusted_limit = requested_limit  
            else:
                adjusted_limit = requested_limit
                print(f"\n Requested comment limit ({requested_limit}) is reasonable.")
                print(f"   System will stream comments and monitor rate limits.")
            
            rate_limit_info['adjusted_limit'] = adjusted_limit
            rate_limit_info['comment_streaming_mode'] = True
        else:
            
            print(f"   Reddit API Maximum Listing Limit: {REDDIT_MAX_LISTING_LIMIT}")
            
            
            if remaining_requests is not None:
                if remaining_requests < requested_limit:
                    print(f"\n  WARNING: Remaining requests ({remaining_requests}) is less than requested limit ({requested_limit})")
                    if reset_timestamp:
                        reset_time = datetime.fromtimestamp(reset_timestamp)
                        print(f"   Rate limit will refresh at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   The system will process up to {remaining_requests} items or wait for limit refresh.")
                    
                    adjusted_limit = min(requested_limit, remaining_requests) if remaining_requests > 0 else 0
                else:
                    print(f"\n Sufficient requests available ({remaining_requests} remaining)")
                    adjusted_limit = requested_limit
            else:
                adjusted_limit = requested_limit
            
            
            if adjusted_limit > REDDIT_MAX_LISTING_LIMIT:
                adjusted_limit = REDDIT_MAX_LISTING_LIMIT
                rate_limit_info['adjusted_limit'] = adjusted_limit
                print(f"\n  WARNING: Requested limit ({requested_limit}) exceeds Reddit API maximum ({REDDIT_MAX_LISTING_LIMIT})")
                print(f" Adjusted limit to maximum possible: {adjusted_limit}")
            else:
                rate_limit_info['adjusted_limit'] = adjusted_limit
                if adjusted_limit == requested_limit:
                    print(f"\n Requested limit ({requested_limit}) is within possible limit ({REDDIT_MAX_LISTING_LIMIT})")
                    print(f" Using requested limit: {adjusted_limit}")
                else:
                    print(f"\n Adjusted limit to {adjusted_limit} (based on remaining requests and API limits)")
        
        print("="*70 + "\n")
        
        return adjusted_limit, rate_limit_info
        
    except Exception as e:
        print(f"\n  Error checking rate limits: {e}")
        print(f"Using requested limit: {requested_limit}")
        print("="*70 + "\n")
        rate_limit_info['adjusted_limit'] = requested_limit
        return adjusted_limit, rate_limit_info

def print_rate_limit_headers(r, context=""):
    
    try:
        if hasattr(r, 'auth') and hasattr(r.auth, 'limits'):
            limits = r.auth.limits
            if limits:
                remaining = limits.get('remaining', 'N/A')
                used = limits.get('used', 'N/A')
                reset_timestamp = limits.get('reset_timestamp', None)
                
                print(f"\n{'='*70}")
                print(f" REDDIT API RATE LIMIT STATUS {context}")
                print(f"{'='*70}")
                print(f"   Requests Used: {used}")
                print(f"   Requests Remaining (Pending): {remaining}")
                
                if reset_timestamp:
                    reset_time = datetime.fromtimestamp(reset_timestamp)
                    reset_time_str = reset_time.strftime('%Y-%m-%d %H:%M:%S')
                    time_until_reset = reset_time - datetime.now()
                    minutes_until_reset = int(time_until_reset.total_seconds() / 60)
                    
                    print(f"   Rate Limit Resets At: {reset_time_str}")
                    print(f"   Time Until Reset: {minutes_until_reset} minutes")
                else:
                    print(f"   Rate Limit Reset Time: Not available")
                
                print(f"{'='*70}")
                
                
                if isinstance(remaining, (int, float)):
                    if remaining <= 0:
                        print(f"\n  CRITICAL: No requests remaining! Rate limit exhausted.")
                        if reset_timestamp:
                            reset_time = datetime.fromtimestamp(reset_timestamp)
                            print(f"   Limit will refresh at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    elif remaining < 5:
                        print(f"\n  CRITICAL: Rate limit very low! Only {remaining} requests remaining.")
                        if reset_timestamp:
                            reset_time = datetime.fromtimestamp(reset_timestamp)
                            print(f"   Limit will refresh at {reset_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    elif remaining < 10:
                        print(f"\n  WARNING: Rate limit is running low! Only {remaining} requests remaining.")
                
                print()  
                
                return {
                    'remaining': remaining,
                    'used': used,
                    'reset_timestamp': reset_timestamp
                }
    except Exception as e:
        print(f"  Warning: Could not access rate limit info: {e}")
    
    return None

@app.route('/', methods=['POST', 'GET'])
def get_data():
    """Handle POST requests with JSON config or GET requests"""
    if request.method == 'GET':
        return jsonify({
            'message': 'Producer service is running',
            'endpoint': 'POST / with JSON config to fetch Reddit data'
        }), 200
    
    
    if not request.data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        config = loads(request.data.decode('utf-8'))
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    data = {}
    
    r = praw.Reddit(
        user_agent=Config.REDDIT_USER_AGENT,
        client_id=Config.REDDIT_CLIENT_ID,
        client_secret=Config.REDDIT_CLIENT_SECRET
    )
    
    subreddit_name = config['subreddit']
    requested_limit = config['limit']
    is_comment_mode = config.get('comments', False)
    
    
    adjusted_limit, rate_limit_info = check_reddit_rate_limits(r, subreddit_name, requested_limit, is_comment_mode=is_comment_mode)
    limit = adjusted_limit
    
    
    config_keys_to_remove = ['subreddit', 'limit', 'data_file']
    for key in config_keys_to_remove:
        config.pop(key, None)

   
    if is_comment_mode:
        
        submissions = r.subreddit(subreddit_name).hot(limit=submission_limit)
        total_comments_to_fetch = limit  
        print(f"Note: Processing up to {submission_limit} submissions to extract up to {total_comments_to_fetch} comments")
    else:
        submissions = r.subreddit(subreddit_name).hot(limit=limit)
        total_comments_to_fetch = limit
    
    comment_count = 0
    submissions_processed = 0
    
    if config.get('comments'):
        
        print("\n" + "="*70)
        print("STARTING COMMENT STREAMING FROM REDDIT")
        print("="*70)
        print(f"Target: {total_comments_to_fetch} comments")
        print(f"Subreddit: r/{subreddit_name}")
        print("="*70 + "\n")
        
        
        print_rate_limit_headers(r, "(Before Streaming)")
        
        
        rate_limit_check_interval = 50 
        last_rate_limit_check = 0
        
        for submission in submissions:
            try:
                
                if comment_count - last_rate_limit_check >= rate_limit_check_interval:
                    print_rate_limit_headers(r, f"(During Streaming - {comment_count} comments processed)")
                    last_rate_limit_check = comment_count
                
                
                submission.comments.replace_more(limit=None)
                
                
                submission_data = {
                    'submission_id': submission.id,
                    'submission_title': submission.title,
                    'submission_author': submission.author.name if submission.author else '[deleted]',
                    'submission_created_utc': submission.created_utc,
                    'submission_score': submission.score,
                    'submission_url': submission.url,
                    'submission_permalink': submission.permalink,
                    'submission_num_comments': submission.num_comments,
                    'subreddit': subreddit_name,
                    'submission_selftext': getattr(submission, 'selftext', ''),
                }
                
                
                def extract_comment(comment, depth=0, parent_id=None):
                    
                    nonlocal comment_count
                    
                    
                    if isinstance(comment, praw.models.MoreComments):
                        return
                    
                    if not hasattr(comment, 'body'):
                        return
                    
                    if comment.body == '[deleted]' or comment.body == '[removed]':
                        return
                    
                    
                    if comment_count >= total_comments_to_fetch:
                        return
                    
                    try:
                       
                        comment_record = {
                            'comment_id': comment.id,
                            'comment_body': comment.body,
                            'comment_author': comment.author.name if comment.author else '[deleted]',
                            'comment_created_utc': comment.created_utc,
                            'comment_score': comment.score,
                            'comment_is_submitter': comment.is_submitter,
                            'comment_edited': comment.edited if hasattr(comment, 'edited') else False,
                            'comment_depth': depth,  
                            'comment_parent_id': parent_id,  
                            'comment_submission_id': submission.id,
                            'comment_submission_title': submission.title,
                            'comment_subreddit': subreddit_name,
                            'comment_permalink': f"https://reddit.com{comment.permalink}",
                            'comment_link_id': comment.link_id,
                            'comment_type': 'reply' if depth > 0 else 'top_level',
                            '_extracted_at': datetime.now().isoformat(),
                        }
                        
                        
                        comment_record.update(submission_data)
                        
                        if MONITORING_AVAILABLE:
                            future, msg_key = monitoring_producer.send_with_monitoring(
                                value=comment_record,
                                key=comment.id
                            )
                            future.get()
                        else:
                            standard_producer.send(Config.KAFKA_TOPIC, value=comment_record, key=comment.id.encode('utf-8'))
                        
                        comment_count += 1
                        
                        
                        if hasattr(comment, 'replies') and comment.replies:
                            for reply in comment.replies:
                                extract_comment(reply, depth=depth + 1, parent_id=comment.id)
                                
                                
                                if comment_count >= total_comments_to_fetch:
                                    return
                                    
                    except Exception as e:
                        print(f"Error extracting comment {comment.id}: {e}")
                        return  
                
                
                for comment in submission.comments.list():
                    if comment_count >= total_comments_to_fetch:
                        break
                    extract_comment(comment, depth=0, parent_id=None)
                
                submissions_processed += 1
                
                
                if submissions_processed % 10 == 0:  
                    rate_info = print_rate_limit_headers(r, f"(After {submissions_processed} submissions, {comment_count} comments)")
                    if rate_info and isinstance(rate_info.get('remaining'), (int, float)):
                        if rate_info['remaining'] < 5:
                            print(f"  CRITICAL: Rate limit very low ({rate_info['remaining']} remaining). Consider reducing request frequency.")
                            sleep(2)  
                
                
                sleep(0.5)
                
            except Exception as e:
                print(f"Error processing submission {submission.id}: {e}")
                continue
            
            
            if comment_count >= total_comments_to_fetch:
                break
        
        
        print("\n" + "="*70)
        print("COMMENT STREAMING COMPLETE")
        print("="*70)
        print(f"Total Comments Extracted: {comment_count}")
        print(f"Submissions Processed: {submissions_processed}")
        print_rate_limit_headers(r, "(After Streaming)")
                
    else:
        
        for submission in submissions:
            if submission.id not in data:
                data[submission.id] = {}
            
            for key in config:
                if config[key]:
                    if key == 'author':
                        data[submission.id][key] = vars(submission)[key].name
                    else:
                        data[submission.id][key] = vars(submission)[key]
            
            
            if MONITORING_AVAILABLE:
                future, msg_key = monitoring_producer.send_with_monitoring(
                    value=data[submission.id],
                    key=submission.id
                )
                future.get()
            else:
                standard_producer.send(Config.KAFKA_TOPIC, value=data[submission.id])
            submissions_processed += 1
            sleep(1)

    return jsonify({
        'status': 'OK',
        'comments_extracted': comment_count if config.get('comments') else 0,
        'submissions_processed': submissions_processed,
        'requested_limit': requested_limit,
        'adjusted_limit': adjusted_limit,
        'rate_limit_info': rate_limit_info
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'reddit-producer'}), 200

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get Kafka monitoring metrics"""
    if MONITORING_AVAILABLE:
        return jsonify(monitoring_producer.get_metrics()), 200
    else:
        return jsonify({'error': 'Monitoring not available'}), 503
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PRODUCER_PORT, debug=True)
