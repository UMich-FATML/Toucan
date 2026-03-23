import json
import requests
import uuid
import os
import re
from time import sleep
import ast
import inspect
import types
import asyncio
import base64

# Add MCP imports
try:
    import mcp
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

################
# File I/O
################

# File I/O utilities
def load_jsonl_to_list(jsonl_file_path):
    data_list = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list

# Load dataset
def load_dataset_from_file(filename):
    #if the file is json
    if filename.endswith('.json'):
        with open(filename, 'r') as file:
            return json.load(file)
    elif filename.endswith('.jsonl'):
        return load_jsonl_to_list(filename)
    else:
        raise ValueError("Invalid file format. Please provide a .json or .jsonl file.")

# Get model abbreviation from model_configs.json
def get_model_abbreviation(model_path, config_file="model_configs.json"):
    """
    Get model abbreviation from model_configs.json
    
    Args:
        model_path: Full model path (e.g., "Qwen/Qwen3-32B")
        config_file: Path to model configs file
        
    Returns:
        str: Model abbreviation (e.g., "q332b")
    """
    if not os.path.isabs(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, config_file)

    try:
        with open(config_file, 'r') as f:
            model_configs = json.load(f)
        
        if model_path in model_configs:
            return model_configs[model_path]["abbreviation"]
        else:
            print(f"⚠️  Model {model_path} not found in {config_file}, falling back to get_model_short_name")
            return get_model_short_name(model_path)
    except FileNotFoundError:
        print(f"⚠️  {config_file} not found, falling back to get_model_short_name")
        return get_model_short_name(model_path)
    except Exception as e:
        print(f"⚠️  Error reading {config_file}: {str(e)}, falling back to get_model_short_name")
        return get_model_short_name(model_path)

# Save dataset
def save_dataset(data, filename, convert_to_jsonl=False):
    if convert_to_jsonl:
        with open(filename, 'w') as file:
            for obj in data:
                file.write(json.dumps(obj) + '\n')
    else:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)

# Safe checkpoint saving with backup
def safe_save_checkpoint(data, checkpoint_file, convert_to_jsonl=False):
    """
    Safely save checkpoint data with backup mechanism:
    1. Rename existing checkpoint to _old
    2. Save new checkpoint
    3. Remove old backup
    
    Args:
        data: Data to save
        checkpoint_file: Path to checkpoint file
        convert_to_jsonl: Whether to save as JSONL format
    """
    old_checkpoint = f"{checkpoint_file}_old"
    
    try:
        # Step 1: Backup existing checkpoint if it exists
        if os.path.exists(checkpoint_file):
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)  # Remove any existing old backup
            os.rename(checkpoint_file, old_checkpoint)
            print(f"📦 Backed up existing checkpoint to {old_checkpoint}")
        
        # Step 2: Save new checkpoint
        save_dataset(data, checkpoint_file, convert_to_jsonl=convert_to_jsonl)
        print(f"💾 Saved new checkpoint to {checkpoint_file}")
        
        # Step 3: Clean up old backup
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)
            print(f"🗑️  Removed old backup {old_checkpoint}")
            
    except Exception as e:
        print(f"❌ Error during checkpoint save: {str(e)}")
        
        # Try to restore from backup if new save failed
        if os.path.exists(old_checkpoint) and not os.path.exists(checkpoint_file):
            try:
                os.rename(old_checkpoint, checkpoint_file)
                print(f"🔄 Restored checkpoint from backup")
            except Exception as restore_error:
                print(f"❌ Failed to restore from backup: {str(restore_error)}")
        
        raise e  # Re-raise the original error


# Function to make a single API request with exponential back-off
def make_api_request_with_retry(message, api_params, api_endpoint, api_headers, max_retries=3):
    payload = api_params.copy()
    payload['messages'] = message

    for attempt in range(max_retries):
        try:
            response = requests.post(api_endpoint, json=payload, headers=api_headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            sleep(2 ** attempt)  # Exponential back-off
    
    print("All retry attempts failed.")
    return None

################
# Data Statistics
################

# Get the mode of the dataset
def parse_usage_number(usage_str, default_value=1):
    """
    Parse usage number from string format, handling cases like '1,234' and missing/zero values.
    Returns default_value for missing or zero usage.
    """
    if not usage_str or usage_str == '0':
        return default_value
    
    # Handle string format like '1,234'
    if isinstance(usage_str, str):
        # Remove commas and convert to int
        try:
            cleaned = usage_str.replace(',', '').strip()
            if not cleaned or cleaned == '0':
                return default_value
            return max(default_value, int(cleaned))
        except (ValueError, AttributeError):
            return default_value
    
    # Handle numeric values
    try:
        return max(default_value, int(usage_str))
    except (ValueError, TypeError):
        return default_value

# Predetermined categories for label classification
PREDETERMINED_CATEGORIES = {
    "Web Search & Research": [
        "web search", "search", "research", "web research", "information gathering", 
        "data retrieval", "academic research", "web scraping", "crawling", "indexing",
        "search engine", "web intelligence", "information extraction", "web analysis",
        "web search &amp; research", "web analytics"
    ],
    "Browser Automation": [
        "browser automation", "browser", "automation", "web automation", "browser control", "selenium", 
        "puppeteer", "web driver", "page interaction", "automated browsing",
        "web scraping", "browser manipulation", "ui automation", "web scraping & automation",
        "web scraping & data extraction"
    ],
    "Memory Management": [
        "memory", "storage", "data storage", "knowledge base", "note taking",
        "notes", "cache", "persistence", "recall", "remember", "memorization",
        "knowledge management", "data retention", "information storage"
    ],
    "Operating System": [
        "os", "operating system", "system", "file system", "process", "command",
        "shell", "terminal", "system administration", "system management",
        "process management", "system commands", "system operations", "system management"
    ],
    "Data Analysis & Processing": [
        "data analysis", "analytics", "data processing", "statistics", "analysis",
        "data transformation", "statistical analysis", "data science", "metrics",
        "reporting", "visualization", "data mining", "processing", "computation",
        "data analysis &amp; processing", "mathematical computation", "math & calculations",
        "math & symbolic computation", "arithmetic operations", "basic calculation",
        "calculation & mathematics", "math operations"
    ],
    "Cryptocurrency & Blockchain": [
        "crypto", "cryptocurrency", "blockchain", "bitcoin", "ethereum", "defi",
        "trading", "wallet", "nft", "token", "smart contract", "web3", "dex",
        "mining", "staking", "digital currency", "chain", "ledger", "web3 development",
        "blockchain development", "blockchain data querying", "blockchain data query"
    ],
    "Daily Productivity": [
        "productivity", "task management", "todo", "scheduling", "organization",
        "planning", "personal", "daily", "routine", "workflow", "efficiency",
        "time management", "personal organization", "task tracking"
    ],
    "File Management": [
        "file", "files", "document", "folder", "directory", "file operations",
        "file handling", "document management", "file system", "storage management",
        "file processing", "document handling", "archive", "backup", "document processing",
        "document security"
    ],
    "Database Operations": [
        "database", "db", "sql", "query", "data", "mysql", "postgresql", "mongodb",
        "nosql", "crud", "schema", "table", "record", "database management",
        "data querying", "database operations"
    ],
    "API Integration": [
        "api", "integration", "webhook", "service", "third party", "external",
        "rest", "graphql", "endpoint", "connector", "service integration",
        "api client", "http", "microservice", "service mesh", "api management"
    ],
    "Communication Tools": [
        "communication", "messaging", "chat", "email", "notification", "message",
        "social", "contact", "discussion", "conversation", "collaboration",
        "team communication", "instant messaging", "mail", "email management",
        "email automation", "email marketing", "email outreach"
    ],
    "Development Tools": [
        "development", "dev", "code", "programming", "developer", "git", "github",
        "version control", "ci/cd", "testing", "debugging", "build", "deployment",
        "software development", "coding", "repo", "repository", "developer tools",
        "version control", "version control systems", "version control & repository management",
        "github integration", "github management", "code analysis", "web development",
        "web app development", "game development"
    ],
    "Security & Authentication": [
        "security", "auth", "authentication", "authorization", "password", "encryption",
        "access control", "oauth", "login", "credentials", "certificate", "ssl",
        "tls", "identity", "permission", "secure", "cybersecurity", "security & authentication",
        "security & analysis", "security & monitoring", "security & surveillance",
        "security & network operations", "security & operations", "network security",
        "cybersecurity & threat intelligence", "cybersecurity & investigation",
        "identity & access management"
    ],
    "Cloud Services": [
        "cloud", "aws", "azure", "gcp", "serverless", "lambda", "function",
        "cloud platform", "cloud computing", "infrastructure", "paas", "saas",
        "iaas", "cloud storage", "cloud deployment"
    ],
    "AI/ML Tools": [
        "ai", "ml", "machine learning", "artificial intelligence", "model",
        "neural network", "deep learning", "nlp", "computer vision", "llm",
        "gpt", "openai", "huggingface", "tensorflow", "pytorch", "prediction",
        "ai/ml tools", "artificial intelligence & problem solving"
    ],
    "Content Creation": [
        "content", "creation", "writing", "editing", "publishing", "blog",
        "article", "text generation", "content management", "cms", "authoring",
        "documentation", "creative writing", "media generation", "content management",
        "content management system", "documentation management"
    ],
    "Social Media": [
        "social media", "twitter", "facebook", "instagram", "linkedin", "youtube",
        "social", "posting", "social platform", "social analytics", "hashtag",
        "social networking", "community", "social engagement", "discord automation",
        "twitter spaces management"
    ],
    "Financial Services": [
        "financial", "finance", "banking", "payment", "money", "accounting",
        "invoice", "billing", "financial data", "budget", "expense", "income",
        "fintech", "payroll", "transaction", "revenue"
    ],
    "E-commerce": [
        "ecommerce", "e-commerce", "shopping", "store", "product", "cart",
        "checkout", "order", "inventory", "retail", "marketplace", "commerce",
        "sales", "customer", "merchant", "payment processing"
    ],
    "Gaming": [
        "gaming", "game", "entertainment", "interactive", "player", "score",
        "leaderboard", "arcade", "puzzle", "rpg", "strategy", "simulation",
        "multiplayer", "esports", "video game", "entertainment", "sports", "sports analytics",
        "sports data & analytics", "sports & racing"
    ],
    "Education": [
        "education", "learning", "course", "student", "teacher", "academic",
        "training", "tutorial", "lesson", "curriculum", "educational",
        "knowledge", "study", "school", "university", "learning management", "educational"
    ],
    "Health & Fitness": [
        "health", "fitness", "medical", "healthcare", "wellness", "exercise",
        "workout", "diet", "nutrition", "medicine", "doctor", "patient",
        "symptom", "diagnosis", "therapy", "mental health", "health & medical calculations",
        "healthcare & pharmaceutical data"
    ],
    "Travel & Maps": [
        "travel", "map", "location", "gps", "navigation", "route", "direction",
        "geography", "destination", "trip", "journey", "tourism", "hotel",
        "flight", "transportation", "geolocation", "geospatial", "geospatial tools",
        "geospatial & mapping", "geospatial data processing", "address & geolocation",
        "navigation & traffic", "transportation & navigation"
    ],
    "News & Media": [
        "news", "media", "journalism", "article", "publication", "press",
        "newsletter", "magazine", "newspaper", "broadcasting", "reporter",
        "editorial", "media consumption", "current events", "media & entertainment",
        "media management", "media retrieval", "music & audio", "audio processing",
        "video processing", "multimedia processing", "media processing"
    ],
    "Weather": [
        "weather", "forecast", "climate", "temperature", "rain", "snow",
        "storm", "meteorology", "atmospheric", "precipitation", "humidity",
        "weather data", "weather monitoring", "climate information", "environmental monitoring"
    ],
    "Time & Calendar": [
        "time", "calendar", "schedule", "appointment", "event", "date",
        "timestamp", "clock", "timezone", "meeting", "reminder", "booking",
        "scheduling", "time management", "calendar integration", "calendar management"
    ],
    "Project Management": [
        "project management"
    ]
}

def normalize_label_for_matching(label):
    """Normalize a label for case-insensitive matching"""
    return label.lower().strip()

def find_matching_category(label):
    """
    Find if a label matches any predetermined category.
    Returns the category name if found, None otherwise.
    """
    if not label:
        return None
    
    normalized_label = normalize_label_for_matching(label)
    
    # First try exact match with category names
    for category in PREDETERMINED_CATEGORIES:
        if normalized_label == normalize_label_for_matching(category):
            return category
    
    # Then try matching with variations
    for category, variations in PREDETERMINED_CATEGORIES.items():
        for variation in variations:
            if normalized_label == normalize_label_for_matching(variation):
                return category
            # Also check if the variation is contained in the label
            if normalize_label_for_matching(variation) in normalized_label:
                return category
    
    return None

################
# Model Abbreviation
################

def get_model_short_name(model_path):
    # Remove org name if present (e.g., "Qwen/Qwen3-32B" -> "Qwen3-32B")
    if "/" in model_path:
        return model_path.split("/")[-1]
    return model_path

################
# API Validation
################

def check_if_api_key_is_valid(profile, smithery_api_key):
    """
    Check if a profile and Smithery API key combination is valid by making a test request to Smithery API.
    Uses a simple HTTP request approach to avoid async complications.
    
    Args:
        profile (str): The profile name to test
        smithery_api_key (str): The Smithery API key to test
    
    Returns:
        dict: {"valid": bool, "message": str, "tools": list or None}
    """
    try:
        # Create configuration
        config = {"debug": False}
        config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
        
        # Create server URL - testing the Smithery API key from the pool
        url = f"https://server.smithery.ai/exa/mcp?config={config_b64}&api_key={smithery_api_key}&profile={profile}"
        
        # Simple HTTP test to check if the endpoint is accessible
        import requests
        try:
            # Test with a simple GET request first to see if the endpoint responds
            response = requests.get(url, timeout=10)
            
            # If we get any response (even an error), it means the API key format is likely correct
            # and the service is accessible. For more detailed validation, we'd need MCP.
            if response.status_code == 200:
                return {
                    "valid": True,
                    "message": "Successfully connected via HTTP. Endpoint accessible.",
                    "tools": ["http_validated"]  # Simple indicator
                }
            elif response.status_code == 401:
                return {
                    "valid": False,
                    "message": "Authentication failed. Invalid API key or profile.",
                    "tools": None
                }
            elif response.status_code == 403:
                return {
                    "valid": False,
                    "message": "Access forbidden. Check API key permissions.",
                    "tools": None
                }
            elif response.status_code == 404:
                return {
                    "valid": False,
                    "message": "Endpoint not found. Check profile name.",
                    "tools": None
                }
            else:
                return {
                    "valid": True,
                    "message": f"Endpoint accessible (HTTP {response.status_code}).",
                    "tools": ["http_validated"]
                }
                
        except requests.exceptions.Timeout:
            return {
                "valid": False,
                "message": "Connection timeout. Service may be down.",
                "tools": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "valid": False,
                "message": "Connection failed. Network or service issue.",
                "tools": None
            }
        except requests.exceptions.RequestException as e:
            return {
                "valid": False,
                "message": f"HTTP request failed: {str(e)}",
                "tools": None
            }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"Error during API validation: {str(e)}",
            "tools": None
        }

def validate_api_pool_entry(entry):
    """
    Validate a single entry from the API pool.
    
    Args:
        entry (dict): Entry with 'profile' and 'api_key' fields
    
    Returns:
        dict: Validation result with additional entry info
    """
    if not isinstance(entry, dict) or 'profile' not in entry or 'api_key' not in entry:
        return {
            "valid": False,
            "message": "Invalid entry format. Must have 'profile' and 'api_key' fields.",
            "profile": entry.get('profile', 'unknown'),
            "source": entry.get('source', 'unknown'),
            "tools": None
        }
    
    # Test the Smithery API key from the entry
    result = check_if_api_key_is_valid(entry['profile'], entry['api_key'])
    
    # Add entry metadata to result
    result['profile'] = entry['profile']
    result['source'] = entry.get('source', 'unknown')
    
    return result

def validate_api_pool_from_file(file_path):
    """
    Validate all entries in an API pool JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing API pool
    
    Returns:
        dict: Summary of validation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'api_pool' not in data:
            return {
                "error": "File must contain 'api_pool' key with list of entries",
                "results": []
            }
        
        results = []
        valid_count = 0
        
        for i, entry in enumerate(data['api_pool']):
            print(f"Validating entry {i+1}/{len(data['api_pool'])}: {entry.get('profile', 'unknown')}")
            
            result = validate_api_pool_entry(entry)
            results.append(result)
            
            if result['valid']:
                valid_count += 1
                print(f"  ✓ Valid - {result['message']}")
            else:
                print(f"  ✗ Invalid - {result['message']}")
        
        return {
            "total_entries": len(data['api_pool']),
            "valid_entries": valid_count,
            "invalid_entries": len(data['api_pool']) - valid_count,
            "results": results
        }
        
    except Exception as e:
        return {
            "error": f"Failed to process file: {str(e)}",
            "results": []
        }

################
# Data Cleaning
################

def clean_unusual_line_terminators(text):
    """
    Remove unusual line terminator characters like Line Separator (U+2028) 
    and Paragraph Separator (U+2029) that can cause issues.
    """
    if isinstance(text, str):
        # Remove Line Separator (U+2028) and Paragraph Separator (U+2029)
        text = text.replace('\u2028', ' ').replace('\u2029', ' ')
        # Also clean other problematic Unicode characters
        text = text.replace('\u0085', ' ')  # Next Line (NEL)
        text = text.replace('\u000B', ' ')  # Vertical Tab
        text = text.replace('\u000C', ' ')  # Form Feed
    return text

def clean_json_object(obj):
    """
    Recursively clean unusual line terminators from a JSON object.
    """
    if isinstance(obj, str):
        return clean_unusual_line_terminators(obj)
    elif isinstance(obj, dict):
        return {key: clean_json_object(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_object(item) for item in obj]
    else:
        return obj

def clean_html_comments(text):
    """
    Remove HTML comments from text content.
    Removes patterns like <!-- followed by any content (non-greedy) and -->
    """
    if not text:
        return text
    
    # Remove HTML comments using regex
    # This pattern matches <!-- followed by any content (non-greedy) and -->
    cleaned_text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Clean up any extra whitespace that might be left
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

################
# Data Preview
################

def create_preview_json(input_file, output_file, num_entries=5):
    """
    Create a preview JSON file with the first N entries from a JSONL file.
    """
    preview_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_entries:
                    break
                preview_data.append(json.loads(line))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(preview_data, f, ensure_ascii=False, indent=2)
        
        print(f"Preview with {len(preview_data)} entries saved to {output_file}")
    except Exception as e:
        print(f"Error creating preview for {input_file}: {e}")

################
# Evaluation Utilities
################

def load_prompt_template(template_path):
    """Load a prompt template from a markdown file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def derive_answer_key_path(input_path):
    """
    Find the *_answer_key.jsonl file in the same directory as the input file.
    Expects exactly one answer key per directory (standard pipeline convention).
    Returns None if not found.
    """
    import glob as glob_mod
    dir_name = os.path.dirname(input_path) or "."
    matches = glob_mod.glob(os.path.join(dir_name, "*_answer_key.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} answer key files in {dir_name}, using first match.")
        return sorted(matches)[0]
    return None

def extract_raw_error(messages):
    """
    Detect if a scenario failed catastrophically.
    Returns: (is_failed: bool, raw_error_string: str or None)

    Checks (in order):
      1. Empty trajectory
      2. Wrapper-caught exceptions ([ERROR: ...], [CRITICAL ERROR: ...], etc.)
      3. Implicit max-turns exceeded (trailing tool call with no follow-up)
      4. In-context tool errors (tool not found / unknown tool)
    """
    if not messages:
        return True, "EMPTY_TRAJECTORY"

    last_msg = messages[-1]
    last_content = str(last_msg.get('content', ''))

    # 1. Wrapper-caught exceptions
    if "[ERROR:" in last_content or "[CRITICAL ERROR:" in last_content or "[UNEXPECTED_ERROR:" in last_content:
        return True, last_content.strip()

    # 2. Trailing tool call → max turns
    if last_msg.get('role') == 'assistant' and ('tool_calls' in last_msg or 'function_call' in last_msg):
        return True, "MAX_TURNS_EXCEEDED (Trailing Tool Call)"

    # 3. In-context tool errors
    for msg in messages:
        if msg.get('role') in ('tool', 'function', 'system'):
            content = str(msg.get('content', '')).lower()
            if "tool not found" in content or "unknown tool" in content or "function not found" in content:
                return True, f"TOOL_ERROR: {msg.get('content')}"

    return False, None

def condense_trajectory(messages, tool_args_limit=500, tool_output_limit=1000):
    """
    Create a condensed text representation of an agent trajectory.

    Includes:
      - User messages (in full)
      - Assistant tool calls (name + truncated arguments)
      - Tool/function outputs (name + truncated content)
      - Assistant text messages

    Skips system messages.
    """
    parts = []
    for msg in messages:
        role = msg.get('role', '')

        if role == 'system':
            continue

        elif role == 'user':
            parts.append(f"[User]: {msg.get('content', '')}")

        elif role == 'assistant':
            # Handle tool_calls (OpenAI format)
            if 'tool_calls' in msg and msg['tool_calls']:
                for tc in msg['tool_calls']:
                    func = tc.get('function', {})
                    name = func.get('name', 'unknown')
                    args_raw = func.get('arguments', '')
                    if isinstance(args_raw, dict):
                        args_raw = json.dumps(args_raw)
                    args_display = args_raw[:tool_args_limit]
                    if len(args_raw) > tool_args_limit:
                        args_display += " ... (truncated)"
                    parts.append(f"[Assistant → Tool Call]: {name}({args_display})")
            # Handle legacy function_call
            elif 'function_call' in msg and msg['function_call']:
                func = msg['function_call']
                name = func.get('name', 'unknown')
                args_raw = func.get('arguments', '')
                if isinstance(args_raw, dict):
                    args_raw = json.dumps(args_raw)
                args_display = args_raw[:tool_args_limit]
                if len(args_raw) > tool_args_limit:
                    args_display += " ... (truncated)"
                parts.append(f"[Assistant → Tool Call]: {name}({args_display})")

            # Assistant text content
            content = msg.get('content', '')
            if content:
                parts.append(f"[Assistant]: {content}")

        elif role in ('tool', 'function'):
            name = msg.get('name', 'unknown')
            content = str(msg.get('content', ''))
            content_display = content[:tool_output_limit]
            if len(content) > tool_output_limit:
                content_display += " ... (truncated)"
            parts.append(f"[Tool Output — {name}]: {content_display}")

    return "\n".join(parts)

def extract_final_response(messages):
    """
    Return the text content of the last assistant message that contains text
    (i.e., not a bare tool-call-only message).
    Returns empty string if no such message exists.
    """
    for msg in reversed(messages):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if content and content.strip():
                return content.strip()
    return ""

def extract_tool_evidence(messages):
    """
    Extract all tool-call / tool-output pairs from a trajectory.

    Returns a list of dicts:
      [{"tool_name": str, "arguments_summary": str, "output": str}, ...]

    Pairs are built by matching tool_call_ids where available,
    otherwise by sequential ordering.
    """
    # First pass: collect all tool calls with their IDs
    pending_calls = {}  # tool_call_id -> {tool_name, arguments_summary}
    ordered_calls = []  # fallback for legacy format without IDs

    for msg in messages:
        if msg.get('role') == 'assistant':
            # OpenAI tool_calls format
            if 'tool_calls' in msg and msg['tool_calls']:
                for tc in msg['tool_calls']:
                    func = tc.get('function', {})
                    name = func.get('name', 'unknown')
                    args_raw = func.get('arguments', '')
                    if isinstance(args_raw, dict):
                        args_raw = json.dumps(args_raw)
                    tc_id = tc.get('id')
                    entry = {"tool_name": name, "arguments_summary": args_raw, "output": ""}
                    if tc_id:
                        pending_calls[tc_id] = entry
                    ordered_calls.append(entry)

            # Legacy function_call format
            elif 'function_call' in msg and msg['function_call']:
                func = msg['function_call']
                name = func.get('name', 'unknown')
                args_raw = func.get('arguments', '')
                if isinstance(args_raw, dict):
                    args_raw = json.dumps(args_raw)
                entry = {"tool_name": name, "arguments_summary": args_raw, "output": ""}
                ordered_calls.append(entry)

        # Match tool outputs back to their calls
        elif msg.get('role') == 'tool':
            tc_id = msg.get('tool_call_id')
            content = str(msg.get('content', ''))
            if tc_id and tc_id in pending_calls:
                pending_calls[tc_id]['output'] = content
            else:
                # Fallback: attach to the first call that still has no output
                for entry in ordered_calls:
                    if not entry['output']:
                        entry['output'] = content
                        break

        elif msg.get('role') == 'function':
            content = str(msg.get('content', ''))
            for entry in ordered_calls:
                if not entry['output']:
                    entry['output'] = content
                    break

    return ordered_calls

def extract_assistant_content(messages):
    """
    Concatenate the text content of ALL assistant messages in a trajectory
    (excluding pure tool-call-only messages that have no text).

    Returns a single string with messages separated by newlines.
    """
    parts = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if content and content.strip():
                parts.append(content.strip())
    return "\n\n".join(parts)
