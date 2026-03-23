import json
import copy
from transformers import AutoTokenizer

# Download here: https://github.com/LLM360/bbq-chat-template/tree/main/bbq-mixed-tool-format
DEFAULT_TOKENIZER_PATH = "/mnt/weka/shrd/k2m/yuekai.sun/bbq-chat-template/bbq-mixed-tool-format"
TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH)

BAD_WORDS = ["chatgpt"]

THINK_KEYS = {'high': 'think', 'medium': 'think_fast', 'low': 'think_faster'}


def ensure_python(obj):
    """Recursively deserialize JSON strings into Python objects."""
    if isinstance(obj, (str, bytes, bytearray)):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return obj
        return ensure_python(obj)
    if isinstance(obj, dict):
        return {k: ensure_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_python(v) for v in obj]
    return obj


def apply_think_key(entry, think_key):
    """Rename think field to the appropriate key for the reasoning effort level."""
    for turn in entry['conversation']:
        if turn['role'] == 'assistant':
            if think_key != 'think':
                turn[think_key] = turn.pop('think', '')
            elif 'think' not in turn:
                turn['think'] = ''
    return entry


def _count_answer_tokens(conversation, think_key):
    """Count tokens in assistant content and tool_calls (excluding think)."""
    answer_tokens = 0
    for turn in conversation:
        if turn['role'] == 'assistant':
            content = turn.get('content', '')
            if content:
                answer_tokens += len(TOKENIZER.encode(content))
            if 'tool_calls' in turn:
                for tc in turn['tool_calls']:
                    rendered = TOKENIZER.apply_chat_template(
                        [{'role': 'assistant', 'tool_calls': [tc]}], tokenize=False
                    )
                    # Strip the wrapper added by chat template
                    inner = rendered.split('<|im_start|>assistant\n')[-1]
                    if inner.endswith('<|im_end|>'):
                        inner = inner[:-len('<|im_end|>')]
                    inner = inner.strip()
                    if inner:
                        answer_tokens += len(TOKENIZER.encode(inner))
    return answer_tokens


def _count_think_tokens(conversation, think_key):
    """Count tokens in the think field of assistant turns."""
    think_tokens = 0
    for turn in conversation:
        if turn['role'] == 'assistant':
            think_content = turn.get(think_key, '')
            if think_content:
                think_tokens += len(TOKENIZER.encode(think_content))
    return think_tokens


def process_entry(entry, reasoning_effort='high', identity_check=True, compute_token_counts=True):
    """Validate, transform, and compute token counts for an SFT dataset entry.

    Returns (processed_entry, errors) -- entry is None if errors is non-empty.

    Always runs apply_chat_template for validation.
    When compute_token_counts=True, also computes token_count, token_count_answer, token_count_think.
    When compute_token_counts=False, skips token counting but still validates via chat template.
    """
    entry = copy.deepcopy(entry)
    errors = []

    # 1. Normalize: rename messages -> conversation
    if 'conversation' not in entry and 'messages' in entry:
        entry['conversation'] = entry.pop('messages')

    if 'conversation' not in entry:
        return (None, ['Conversation is missing'])

    conversation = entry['conversation']

    # 2. Clean empty system message
    if conversation and conversation[0]['role'] == 'system':
        if 'content' in conversation[0] and not conversation[0]['content']:
            del conversation[0]['content']
        if 'tools' in conversation[0] and not conversation[0]['tools']:
            del conversation[0]['tools']
        if 'content' not in conversation[0] and 'tools' not in conversation[0]:
            del conversation[0]

    # 3. Validate roles and fix each turn
    for i, turn in enumerate(conversation):

        ## Delete empty fields
        for key in ['content', 'think', 'tool_calls', 'tools']:
            if key in turn and turn[key] == None:
                del turn[key]

        if i == 0:
            if turn['role'] not in ('system', 'user'):
                errors.append(f'First message must be system or user; got {turn["role"]}')
        else:
            if turn['role'] not in ('user', 'assistant', 'tool'):
                errors.append(f'Turn {i} has invalid role {turn["role"]}; expected user, assistant, or tool')

        if turn['role'] == 'system':
            if 'tools' in turn:
                for j, tool in enumerate(turn['tools']):
                    turn['tools'][j] = ensure_python(tool)

        if turn['role'] == 'assistant':
            if i < len(conversation) - 1 and conversation[i + 1]['role'] not in ['user', 'tool']:
                errors.append(
                    f'Assistant response must be followed by a user or tool message; instead got {conversation[i + 1]["role"]}'
                )

            if not (turn.get('tool_calls', False) or turn.get('content', False)):
                errors.append('Assistant response must have either tools or content - MISSING BOTH')
            elif 'tool_calls' in turn:
                if turn['tool_calls']:
                    turn['tool_calls'] = ensure_python(turn['tool_calls'])
                    # Unwrap nemotron function wrapper
                    for j, tool in enumerate(turn['tool_calls']):
                        if 'function' in tool:
                            turn['tool_calls'][j] = tool['function']
                    if not all('arguments' in tool for tool in turn['tool_calls']):
                        errors.append('Tool calls must have arguments')
                else:
                    del turn['tool_calls']

        if turn['role'] == 'user':
            if i < len(conversation) - 1 and conversation[i + 1]['role'] != 'assistant':
                errors.append(
                    f'User message must be followed by an assistant message; instead got {conversation[i + 1]["role"]}'
                )

    # 4. Tool system validation
    has_tools = any(
        turn['role'] == 'tool' or 'tool_calls' in turn or 'tools' in turn
        for turn in conversation
    )
    if has_tools:
        system_turns = [t for t in conversation if t['role'] == 'system']
        if not system_turns or not any('tools' in t for t in system_turns):
            errors.append('Conversation has tool usage but system message does not declare tools')
        for turn in conversation:
            if turn['role'] == 'system' and 'tools' in turn:
                # Unwrap nemotron function wrapper in system tools
                for j, tool in enumerate(turn['tools']):
                    if 'function' in tool:
                        turn['tools'][j] = tool['function']
                # Fix arguments -> parameters
                for tool in turn['tools']:
                    if 'arguments' in tool:
                        tool['parameters'] = tool.pop('arguments')
                if not all('parameters' in tool for tool in turn['tools']):
                    errors.append('System message must have parameters for all tools')
    # 5. Conversation must end with assistant or a training user message
    if conversation[-1]['role'] not in ('assistant', 'user'):
        errors.append(
            f'Conversation must end with an assistant or user message; ended with {conversation[-1]["role"]}'
        )

    if errors:
        return (None, errors)

    # 6. Apply think key
    think_key = THINK_KEYS[reasoning_effort]
    apply_think_key(entry, think_key)

    # 7. Chat template validation
    try:
        txt = TOKENIZER.apply_chat_template(conversation, tokenize=False)
    except Exception as e:
        return (None, [f'Failed to apply chat template: {e}'])

    # 8. Token counting
    if compute_token_counts:
        entry['token_count'] = len(TOKENIZER.encode(txt))
        entry['token_count_answer'] = _count_answer_tokens(conversation, think_key)
        entry['token_count_think'] = _count_think_tokens(conversation, think_key)
    else:
        # Validate that counts already exist
        for field in ['token_count', 'token_count_answer', 'token_count_think']:
            if entry.get(field) is None:
                errors.append(f'{field} is missing (compute_token_counts=False)')

    if errors:
        return (None, errors)

    # 9. Identity check
    if identity_check:
        all_text_parts = []
        for turn in conversation:
            all_text_parts.append(turn.get('content', ''))
            all_text_parts.append(turn.get(think_key, ''))
        all_text = ' '.join(all_text_parts).lower()
        for word in BAD_WORDS:
            if word in all_text:
                errors.append(f'Conversation contains bad word: {word}')
                break

    if errors:
        return (None, errors)

    return (entry, [])


if __name__ == '__main__':
    import os

    passed = 0
    failed = 0

    def run_test(name, condition):
        global passed, failed
        if condition:
            print(f'  [PASS] {name}')
            passed += 1
        else:
            print(f'  [FAIL] {name}')
            failed += 1

    # Test 1: Valid entry with think
    print('Test 1: Valid entry with think')
    entry1 = {
        'conversation': [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': '4', 'think': 'Simple arithmetic: 2+2=4'},
        ]
    }
    result, errs = process_entry(entry1)
    run_test('no errors', errs == [])
    run_test('result is not None', result is not None)
    run_test('token_count present', 'token_count' in result)
    run_test('token_count_answer present', 'token_count_answer' in result)
    run_test('token_count_think present', 'token_count_think' in result)
    run_test('token_count > 0', result['token_count'] > 0)
    run_test('token_count_answer > 0', result['token_count_answer'] > 0)
    run_test('token_count_think > 0', result['token_count_think'] > 0)
    print()

    # Test 2: reasoning_effort='medium' -> think_fast key
    print('Test 2: reasoning_effort=medium')
    entry2 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!', 'think': 'A greeting'},
        ]
    }
    result2, errs2 = process_entry(entry2, reasoning_effort='medium')
    run_test('no errors', errs2 == [])
    run_test('think_fast key present', 'think_fast' in result2['conversation'][1])
    run_test('think key removed', 'think' not in result2['conversation'][1])
    print()

    # Test 3: Invalid entry (empty assistant)
    print('Test 3: Invalid entry (empty assistant)')
    entry3 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': ''},
        ]
    }
    result3, errs3 = process_entry(entry3)
    run_test('errors returned', len(errs3) > 0)
    run_test('result is None', result3 is None)
    print()

    # Test 4: Bad words entry
    print('Test 4: Bad words identity check')
    entry4 = {
        'conversation': [
            {'role': 'user', 'content': 'Tell me about ChatGPT'},
            {'role': 'assistant', 'content': 'ChatGPT is a language model.', 'think': ''},
        ]
    }
    result4, errs4 = process_entry(entry4, identity_check=True)
    run_test('errors returned', len(errs4) > 0)
    run_test('bad word error', any('bad word' in e for e in errs4))
    # With identity_check disabled
    result4b, errs4b = process_entry(entry4, identity_check=False)
    run_test('no errors with identity_check=False', errs4b == [])
    print()

    # Test 5: messages key instead of conversation
    print('Test 5: messages key normalization')
    entry5 = {
        'messages': [
            {'role': 'user', 'content': 'Test'},
            {'role': 'assistant', 'content': 'Response', 'think': ''},
        ]
    }
    result5, errs5 = process_entry(entry5)
    run_test('no errors', errs5 == [])
    run_test('conversation key in result', 'conversation' in result5)
    run_test('messages key removed', 'messages' not in result5)
    print()

    # Test 6: compute_token_counts=False with pre-existing counts
    print('Test 6: compute_token_counts=False with pre-existing counts')
    entry6 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': ''},
        ],
        'token_count': 100,
        'token_count_answer': 50,
        'token_count_think': 10,
    }
    result6, errs6 = process_entry(entry6, compute_token_counts=False)
    run_test('no errors', errs6 == [])
    run_test('token_count not recomputed', result6['token_count'] == 100)
    run_test('token_count_answer not recomputed', result6['token_count_answer'] == 50)
    run_test('token_count_think not recomputed', result6['token_count_think'] == 10)
    print()

    # Test 6b: compute_token_counts=False without pre-existing counts -> error
    print('Test 6b: compute_token_counts=False without pre-existing counts')
    entry6b = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': ''},
        ],
    }
    result6b, errs6b = process_entry(entry6b, compute_token_counts=False)
    run_test('errors returned', len(errs6b) > 0)
    run_test('result is None', result6b is None)
    print()

    # Test 7: Entry with tool_calls
    print('Test 7: Entry with tool_calls')
    entry7 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': 'Get weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'What is the weather in NYC?'},
            {'role': 'assistant', 'content': '', 'think': 'I should call get_weather',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': '{"temp": 72}'},
            {'role': 'assistant', 'content': 'The temperature in NYC is 72F.', 'think': ''},
        ]
    }
    result7, errs7 = process_entry(entry7)
    run_test('no errors', errs7 == [])
    run_test('token_count_answer > 0', result7['token_count_answer'] > 0)
    run_test('token_count_think > 0', result7['token_count_think'] > 0)
    print()

    # Test 8: Entry ending with user (training user message)
    print('Test 8: Entry ending with user')
    entry8 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!', 'think': ''},
            {'role': 'user', 'content': 'Tell me more'},
        ]
    }
    result8, errs8 = process_entry(entry8)
    run_test('no errors', errs8 == [])
    run_test('result is not None', result8 is not None)
    run_test('token_count present', 'token_count' in result8)
    print()

    # Test 9: Real JSONL entry from processed data (if available)
    print('Test 9: Real JSONL entry (if available)')
    real_data_paths = [
        '/lustrefs/users/mikhail.yurochkin/synthetic_data_m2/sft/PROCESSED/keep/',
        '/lustrefs/users/mikhail.yurochkin/synthetic_data_m2/sft/PROCESSED/cleaned/',
    ]
    real_entry_found = False
    for real_dir in real_data_paths:
        if not os.path.isdir(real_dir):
            continue
        for fname in os.listdir(real_dir):
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(real_dir, fname)
            with open(fpath, 'r') as f:
                line = f.readline().strip()
                if line:
                    real_entry = json.loads(line)
                    real_entry_found = True
                    stored_tc = real_entry.get('token_count')
                    result9, errs9 = process_entry(real_entry)
                    run_test('no errors on real entry', errs9 == [])
                    if result9 and stored_tc is not None:
                        computed_tc = result9['token_count']
                        run_test(
                            f'token_count matches (stored={stored_tc}, computed={computed_tc})',
                            stored_tc == computed_tc,
                        )
                    break
            if real_entry_found:
                break
        if real_entry_found:
            break
    if not real_entry_found:
        print('  [SKIP] No real data found')
    print()

    # Test 10: Empty system message gets cleaned
    print('Test 10: Empty system message cleanup')
    entry10 = {
        'conversation': [
            {'role': 'system', 'content': '', 'tools': []},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': ''},
        ]
    }
    result10, errs10 = process_entry(entry10)
    run_test('no errors', errs10 == [])
    run_test('system message removed', result10['conversation'][0]['role'] == 'user')
    print()

    # Test 11: reasoning_effort='low' -> think_faster key
    print('Test 11: reasoning_effort=low')
    entry11 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'Greeting'},
        ]
    }
    result11, errs11 = process_entry(entry11, reasoning_effort='low')
    run_test('no errors', errs11 == [])
    run_test('think_faster key present', 'think_faster' in result11['conversation'][1])
    run_test('think key removed', 'think' not in result11['conversation'][1])
    print()

    # Test 12: Original entry is not mutated (deepcopy)
    print('Test 12: Original entry not mutated')
    entry12 = {
        'messages': [
            {'role': 'user', 'content': 'Test'},
            {'role': 'assistant', 'content': 'Response', 'think': 'Reasoning'},
        ]
    }
    original_keys = set(entry12.keys())
    process_entry(entry12, reasoning_effort='medium')
    run_test('original entry unchanged', set(entry12.keys()) == original_keys)
    run_test('messages key still present', 'messages' in entry12)
    print()

    # Test 13: Invalid first message role
    print('Test 13: Invalid first message role')
    entry13 = {
        'conversation': [
            {'role': 'assistant', 'content': 'Hi', 'think': ''},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi again', 'think': ''},
        ]
    }
    result13, errs13 = process_entry(entry13)
    run_test('errors returned', len(errs13) > 0)
    run_test('invalid role error', any('First message' in e for e in errs13))
    print()

    # Test 14: Invalid role in subsequent turn
    print('Test 14: Invalid role in subsequent turn')
    entry14 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'system', 'content': 'You are helpful'},
            {'role': 'assistant', 'content': 'Hi', 'think': ''},
        ]
    }
    result14, errs14 = process_entry(entry14)
    run_test('errors returned', len(errs14) > 0)
    run_test('invalid role error', any('invalid role' in e for e in errs14))
    print()

    # Test 15: Tool usage without system tools declaration
    print('Test 15: Tool usage without system tools declaration')
    entry15 = {
        'conversation': [
            {'role': 'user', 'content': 'What is the weather?'},
            {'role': 'assistant', 'content': '', 'think': 'Call tool',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': '{"temp": 72}'},
            {'role': 'assistant', 'content': '72F', 'think': ''},
        ]
    }
    result15, errs15 = process_entry(entry15)
    run_test('errors returned', len(errs15) > 0)
    run_test('missing tools error', any('does not declare tools' in e for e in errs15))
    print()

    # Summary
    total = passed + failed
    print(f'Results: {passed}/{total} passed, {failed}/{total} failed')
    if failed == 0:
        print('All tests passed!')
    else:
        print('Some tests failed!')
        exit(1)
