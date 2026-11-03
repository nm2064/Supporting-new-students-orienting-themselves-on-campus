"""Request-local answer previews; final citation validation remains authoritative."""
from contextvars import ContextVar
from functools import wraps
import json
import re
answer_sink = ContextVar('answer_sink', default=None)
answer_generation = ContextVar('answer_generation', default=False)

def streamed_answer(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = answer_generation.set(True)
        try:
            return fn(*args, **kwargs)
        finally:
            answer_generation.reset(token)
    return wrapped

def partial_answer(text):
    match = re.search(r'"answer_markdown"\s*:\s*"', text)
    if not match:
        return None
    value = text[match.end():]
    # Decode only complete JSON characters, withholding incomplete escapes.
    result, i = '', 0
    while i < len(value):
        if value[i] == '"':
            break
        if value[i] == '\\':
            size = 6 if value[i:i+2] == '\\u' else 2
            if i + size > len(value):
                break
            try:
                result += json.loads('"' + value[i:i+size] + '"')
            except ValueError:
                break
            i += size
        else:
            result += value[i]
            i += 1
    return result
