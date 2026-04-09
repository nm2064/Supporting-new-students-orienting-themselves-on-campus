# Frontend Integration Guide

Complete guide to integrate the Agentic RAG backend with your `index.html`.

## Table of Contents

1. [Quick Start (Just Copy-Paste)](#quick-start)
2. [Complete ChatPage Component](#complete-chatpage-component)
3. [Step-by-Step Changes](#step-by-step-changes)
4. [Optional: Streaming Support](#optional-streaming-support)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option A: Minimal (No Changes Required ✅)

Your existing code already works! The backend returns `{answer: "..."}` which matches what you expect.

### Option B: Enhanced (Recommended)

**Step 1:** Start the backend:
```bash
cd UniBot
python -m venv .venv
.venv\Scripts\activate
pip install -r agentic_rag\requirements.txt
copy agentic_rag\.env.example agentic_rag\.env
python -m agentic_rag
```

**Step 2:** Replace your entire `ChatPage` component (around line 265 in index.html) with the [Complete ChatPage Component](#complete-chatpage-component) below.

---

## Complete ChatPage Component

Replace your entire `ChatPage` component with this enhanced version:

```javascript
const ChatPage = () => {
    // Messages state - now includes sources metadata
    const [messages, setMessages] = useState([
        { 
            role: "assistant", 
            text: "Hi! I'm UniBot, your campus assistant. Ask me about library hours, parking, housing, or anything else!" 
        }
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [expandedSources, setExpandedSources] = useState({});
    const messagesEndRef = React.useRef(null);

    // Session ID helper - persists across page refreshes
    const getSessionId = () => {
        let sessionId = localStorage.getItem('unibot_session_id');
        if (!sessionId) {
            sessionId = typeof crypto !== 'undefined' && crypto.randomUUID 
                ? crypto.randomUUID()
                : `${Date.now()}-${Math.random().toString(36).substr(2, 9)}-${Math.random().toString(36).substr(2, 9)}`;
            localStorage.setItem('unibot_session_id', sessionId);
        }
        return sessionId;
    };

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Send message to backend
    const sendMessage = async () => {
        if (!input.trim() || loading) return;
        const userText = input.trim();
        setInput("");

        // Add user message locally
        setMessages(prev => [...prev, { role: "user", text: userText }]);
        setLoading(true);
        
        try {
            const res = await fetch("http://localhost:8000/rag-chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    question: userText,
                    session_id: getSessionId()
                }),
            });
            
            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }
            
            const data = await res.json();
            
            // Add assistant response with sources
            setMessages(prev => [...prev, { 
                role: "assistant", 
                text: data.answer,
                sources: data.sources || [],
                usedRetrieval: data.used_retrieval,
                isClarifying: false
            }]);
            
            // Handle clarifying question if present
            if (data.clarifying_question) {
                setMessages(prev => [...prev, { 
                    role: "assistant", 
                    text: `🤔 ${data.clarifying_question}`,
                    sources: [],
                    usedRetrieval: false,
                    isClarifying: true
                }]);
            }
            
        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { 
                role: "assistant", 
                text: "Sorry, I encountered an error. Please try again.",
                sources: [],
                usedRetrieval: false
            }]);
        } finally {
            setLoading(false);
        }
    };

    // Handle quick prompt buttons
    const handleQuickPrompt = (prompt) => {
        setInput(prompt);
        // Optional: send immediately
        // setTimeout(() => sendMessage(), 100);
    };

    // Clear chat history
    const clearChat = () => {
        setMessages([{ 
            role: "assistant", 
            text: "Hi! I'm UniBot, your campus assistant. Ask me about library hours, parking, housing, or anything else!" 
        }]);
        localStorage.removeItem('unibot_session_id');
    };

    // Render message with optional sources
    const renderMessage = (msg, idx) => {
        const isUser = msg.role === "user";
        const hasSources = msg.sources && msg.sources.length > 0;
        const isClarifying = msg.isClarifying;
        
        return (
            <div key={idx} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
                <div className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    isUser 
                        ? 'bg-primary text-white rounded-br-md' 
                        : isClarifying
                            ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-gray-900 dark:text-white rounded-bl-md'
                            : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-bl-md'
                }`}>
                    {/* Main message text */}
                    <div className="text-sm whitespace-pre-wrap leading-relaxed">{msg.text}</div>
                    
                    {/* Sources dropdown (only for assistant messages with sources) */}
                    {hasSources && !isUser && (
                        <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
                            <button 
                                onClick={() => setExpandedSources(prev => ({...prev, [idx]: !prev[idx]}))}
                                className="flex items-center gap-1 text-xs text-primary hover:opacity-80 transition-opacity"
                            >
                                <span className="material-symbols-outlined text-sm">
                                    {expandedSources[idx] ? 'expand_less' : 'expand_more'}
                                </span>
                                <span className="font-medium">
                                    {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''} 
                                    {msg.usedRetrieval ? ' (RAG)' : ''}
                                </span>
                            </button>
                            
                            {expandedSources[idx] && (
                                <div className="mt-2 space-y-2">
                                    {msg.sources.map((source, sidx) => (
                                        <div key={sidx} className="text-xs bg-white/50 dark:bg-black/20 rounded-lg p-3">
                                            <div className="flex items-center gap-2">
                                                <span className="bg-primary/10 text-primary px-2 py-0.5 rounded font-medium">
                                                    [{sidx + 1}] {source.chunk_id}
                                                </span>
                                                <span className="text-gray-500">{source.source}</span>
                                            </div>
                                            <div className="text-gray-600 dark:text-gray-400 mt-2 leading-relaxed">
                                                {source.excerpt}
                                            </div>
                                            <div className="text-gray-400 text-[10px] mt-2 flex items-center gap-1">
                                                <span className="material-symbols-outlined text-xs">trending_up</span>
                                                Relevance: {Math.round(source.score * 100)}%
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                    
                    {/* Retrieval indicator (small badge) */}
                    {!hasSources && msg.usedRetrieval && !isUser && !isClarifying && (
                        <div className="mt-2 text-[10px] text-gray-400 flex items-center gap-1">
                            <span className="material-symbols-outlined text-xs">psychology</span>
                            Direct answer
                        </div>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className="bg-background-light dark:bg-background-dark font-display flex justify-center min-h-screen">
            <div className="relative w-full max-w-[450px] bg-white dark:bg-gray-900 shadow-2xl flex flex-col h-screen overflow-hidden border-x border-gray-100 dark:border-gray-800">
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800 bg-white/95 dark:bg-gray-900/95 backdrop-blur-md">
                    <div className="flex items-center gap-3">
                        <div className="size-10 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/30">
                            <span className="material-symbols-outlined">smart_toy</span>
                        </div>
                        <div>
                            <h1 className="font-bold text-gray-900 dark:text-white">UniBot</h1>
                            <div className="flex items-center gap-1 text-xs text-green-600">
                                <span className="size-2 bg-green-500 rounded-full animate-pulse"></span>
                                Online
                            </div>
                        </div>
                    </div>
                    <button 
                        onClick={clearChat}
                        className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                        title="Clear chat"
                    >
                        <span className="material-symbols-outlined">delete_outline</span>
                    </button>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((msg, idx) => renderMessage(msg, idx))}
                    
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-gray-100 dark:bg-gray-800 rounded-2xl rounded-bl-md px-4 py-3">
                                <div className="flex items-center gap-1">
                                    <span className="size-2 bg-primary rounded-full animate-bounce" style={{animationDelay: '0ms'}}></span>
                                    <span className="size-2 bg-primary rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span>
                                    <span className="size-2 bg-primary rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
                                </div>
                            </div>
                        </div>
                    )}
                    
                    <div ref={messagesEndRef} />
                </div>

                {/* Quick Prompts */}
                {messages.length < 3 && (
                    <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-100 dark:border-gray-800">
                        <p className="text-xs text-gray-500 mb-2">Quick questions:</p>
                        <div className="flex gap-2 overflow-x-auto pb-2 hide-scrollbar">
                            {["Library hours?", "Parking fees?", "Housing application?", "Dining options?"].map((prompt) => (
                                <button
                                    key={prompt}
                                    onClick={() => handleQuickPrompt(prompt)}
                                    className="flex-shrink-0 px-3 py-1.5 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-full text-xs text-gray-700 dark:text-gray-300 hover:border-primary hover:text-primary transition-colors"
                                >
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Input */}
                <div className="p-4 border-t border-gray-100 dark:border-gray-800 bg-white dark:bg-gray-900">
                    <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-800 rounded-2xl px-4 py-2">
                        <input
                            className="flex-1 bg-transparent outline-none text-sm text-gray-900 dark:text-white placeholder-gray-400"
                            placeholder="Ask about campus..."
                            type="text"
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                            disabled={loading}
                        />
                        <button
                            className="bg-primary text-white size-10 rounded-xl flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors shadow-lg shadow-primary/30"
                            onClick={sendMessage}
                            disabled={loading || !input.trim()}
                        >
                            <span className="material-symbols-outlined text-xl">send</span>
                        </button>
                    </div>
                </div>

                {/* Bottom Nav */}
                <BottomNav active="chat" />
            </div>
        </div>
    );
};
```

---

## Step-by-Step Changes

If you prefer to make changes manually instead of replacing the whole component:

### Step 1: Add Session Helper (Line ~265)

Add this function inside `ChatPage`:

```javascript
const getSessionId = () => {
    let sessionId = localStorage.getItem('unibot_session_id');
    if (!sessionId) {
        sessionId = typeof crypto !== 'undefined' && crypto.randomUUID 
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        localStorage.setItem('unibot_session_id', sessionId);
    }
    return sessionId;
};
```

### Step 2: Update Fetch Call (Line ~281)

**FROM:**
```javascript
body: JSON.stringify({ question: userText }),
```

**TO:**
```javascript
body: JSON.stringify({ 
    question: userText,
    session_id: getSessionId()
}),
```

### Step 3: Store Sources in Message (Line ~290)

**FROM:**
```javascript
setMessages(prev => [...prev, { role: "assistant", text: data.answer }]);
```

**TO:**
```javascript
setMessages(prev => [...prev, { 
    role: "assistant", 
    text: data.answer,
    sources: data.sources || [],
    usedRetrieval: data.used_retrieval
}]]);
```

---

## Optional: Streaming Support

For a more responsive UI, you can add streaming support. This requires backend changes too.

**Backend endpoint for streaming:**
```python
@app.post("/rag-chat-stream")
async def rag_chat_stream(request: ChatRequest):
    # Implementation returns Server-Sent Events
    ...
```

**Frontend streaming implementation:**
```javascript
const sendMessageStreaming = async () => {
    const res = await fetch("http://localhost:8000/rag-chat-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userText, session_id: getSessionId() }),
    });
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    
    // Add empty assistant message
    setMessages(prev => [...prev, { role: "assistant", text: "" }]);
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        setMessages(prev => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1].text += chunk;
            return newMessages;
        });
    }
};
```

> **Note:** Streaming requires backend modifications. The current backend does not support streaming.

---

## Testing

### Test Queries

| Query | Expected Behavior |
|-------|-------------------|
| `"What are the library hours?"` | Should retrieve sources and show "Sources" dropdown |
| `"Hello"` | Should answer directly, no sources |
| `"Help"` or `"?"` | Should show clarifying question |
| `"What about parking?"` (after asking about library) | Should use session context |

### Check Session Persistence

1. Open browser DevTools → Application → Local Storage
2. Look for `unibot_session_id` key
3. Refresh page - value should remain the same

### Check Sources Display

1. Ask a question about library hours
2. Look for "X sources (RAG)" button below the answer
3. Click to expand and see retrieved chunks

---

## Troubleshooting

### CORS Errors

**Error:**
```
Access to fetch at 'http://localhost:8000/rag-chat' from origin 'null' has been blocked by CORS policy
```

**Solutions:**

1. **Serve HTML through local server (Recommended):**
   ```bash
   # From project root
   python -m http.server 3000
   # Visit http://localhost:3000
   ```

2. **Use VS Code Live Server extension**

3. **Verify CORS is enabled in backend** (`agentic_rag/app.py`):
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       ...
   )
   ```

### Session Not Persisting

**Check:**
- Open DevTools → Application → Local Storage
- Should see `unibot_session_id` with a UUID value

**Fix:**
```javascript
// Make sure this is called BEFORE the fetch
session_id: getSessionId()
```

### No Sources Showing

**Check browser console for response:**
```javascript
const data = await res.json();
console.log(data); // Should show sources array
```

**Expected response:**
```json
{
  "answer": "The library is open...",
  "sources": [
    {
      "source": "knowledge.txt",
      "chunk_id": "chunk_0001",
      "excerpt": "The University Library is open...",
      "score": 0.92
    }
  ],
  "used_retrieval": true
}
```

If `used_retrieval` is `false`, the router decided the question didn't need retrieval.

### Backend Not Responding

**Check:**
```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", "knowledge_file_exists": true, ...}
```

**If not running:**
```bash
cd UniBot
python -m agentic_rag
```

### API Key Errors

**Check backend logs** for:
```
❌ Configuration error: Missing required environment variables
```

**Fix:**
```bash
cd UniBot
copy agentic_rag\.env.example agentic_rag\.env
# Edit agentic_rag\.env and add your keys
```

---

## Summary of Changes Made

| File | Change |
|------|--------|
| `index.html` | Replaced `ChatPage` component with enhanced version |
| Added | Session persistence via `localStorage` |
| Added | Source citations with collapsible UI |
| Added | Loading indicators |
| Added | Quick prompt buttons |
| Added | Clear chat button |
| Added | Auto-scroll to bottom |
| Added | Better error handling |

The enhanced chat UI now shows:
- ✅ Session-based conversation history
- ✅ Source citations for retrieved answers
- ✅ Visual distinction between direct answers and RAG responses
- ✅ Clarifying question UI
- ✅ Loading states
- ✅ Better mobile responsiveness
