"""
Automated evaluation test runner for UniBot.
Sends 10 queries across 5 categories, measures response time, accuracy,
and captures results for the dissertation evaluation section.
"""

import json
import sys
import time
import io

import requests

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:8000"

# Ground-truth test queries with expected information
TEST_QUERIES = [
    # Category 1: Campus Navigation (2 queries)
    {
        "category": "Campus Navigation",
        "query": "Where is the library and what are its opening hours?",
        "expected_keywords": ["library"],
        "session_id": "eval_nav_1",
    },
    {
        "category": "Campus Navigation",
        "query": "How do I get to the James Watt Centre from the Student Union?",
        "expected_keywords": ["james watt", "centre", "building"],
        "session_id": "eval_nav_2",
    },
    # Category 2: Academic Processes (2 queries)
    {
        "category": "Academic Processes",
        "query": "How do I enrol in my modules for the new semester?",
        "expected_keywords": ["enrol", "module", "registration"],
        "session_id": "eval_acad_1",
    },
    {
        "category": "Academic Processes",
        "query": "What support is available if I am struggling with my coursework?",
        "expected_keywords": ["support", "academic", "help"],
        "session_id": "eval_acad_2",
    },
    # Category 3: Societies and Social (2 queries)
    {
        "category": "Societies and Social",
        "query": "What societies can I join at Heriot-Watt?",
        "expected_keywords": ["society", "societies", "club", "union"],
        "session_id": "eval_soc_1",
    },
    {
        "category": "Societies and Social",
        "query": "How can I join the sports union or a sports team?",
        "expected_keywords": ["sport", "team", "union", "gym"],
        "session_id": "eval_soc_2",
    },
    # Category 4: Accommodation (2 queries)
    {
        "category": "Accommodation",
        "query": "What accommodation options are available for first year students?",
        "expected_keywords": ["accommodation", "hall", "residence", "room"],
        "session_id": "eval_acc_1",
    },
    {
        "category": "Accommodation",
        "query": "How do I report a maintenance issue in my student accommodation?",
        "expected_keywords": ["maintenance", "repair", "report", "issue", "accommodation"],
        "session_id": "eval_acc_2",
    },
    # Category 5: Multilingual (2 queries - Chinese and Hindi)
    {
        "category": "Multilingual",
        "query": "图书馆在哪里？开放时间是什么？",  # Where is the library? Opening hours?
        "expected_keywords": ["library", "图书馆"],
        "session_id": "eval_ml_zh",
        "ui_language": "zh-Hans",
    },
    {
        "category": "Multilingual",
        "query": "मैं हेरियट-वाट में कौन सी सोसाइटी में शामिल हो सकता हूँ?",  # What societies at HWU?
        "expected_keywords": ["society", "societies", "सोसाइटी"],
        "session_id": "eval_ml_hi",
        "ui_language": "hi",
    },
]


def run_query(test_case):
    """Send a query to UniBot and measure the response."""
    payload = {
        "message": test_case["query"],
        "session_id": test_case["session_id"],
        "auto_detect": True,
    }
    if "ui_language" in test_case:
        payload["ui_language"] = test_case["ui_language"]

    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/rag-chat", json=payload, timeout=60)
        elapsed = time.time() - start_time
        data = response.json()
    except Exception as e:
        return {
            "error": str(e),
            "elapsed": time.time() - start_time,
        }

    # Check accuracy: does the response contain expected keywords?
    answer = (data.get("answer_markdown") or "").lower()
    answer_combined = answer + " " + (data.get("answer") or "").lower()

    keywords_found = sum(
        1 for kw in test_case["expected_keywords"]
        if kw.lower() in answer_combined
    )
    keyword_ratio = keywords_found / len(test_case["expected_keywords"])

    # Check for citations
    citations = data.get("citations", [])
    sources = data.get("sources", [])
    has_citations = len(citations) > 0

    # Check for hallucination indicators
    used_retrieval = data.get("used_retrieval", False)
    confidence = data.get("confidence", 0)

    return {
        "category": test_case["category"],
        "query": test_case["query"][:60],
        "elapsed_s": round(elapsed, 2),
        "answer_length": len(data.get("answer_markdown", "")),
        "effective_language": data.get("effective_language", "unknown"),
        "used_retrieval": used_retrieval,
        "confidence": round(confidence, 3),
        "num_citations": len(citations),
        "num_sources": len(sources),
        "has_citations": has_citations,
        "keyword_match_ratio": round(keyword_ratio, 2),
        "follow_up_suggestions": len(data.get("follow_up_suggestions", [])),
        "clarifying_question": data.get("clarifying_question") is not None,
        "needs_human_handoff": data.get("needs_human_handoff", False),
        "answer_preview": (data.get("answer_markdown") or "")[:200],
    }


def main():
    print("=" * 70)
    print("UniBot Evaluation Test Runner")
    print("=" * 70)

    # Check health first
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"Status: {health['status']}")
        print(f"Knowledge base: {'loaded' if health['knowledge_file_exists'] else 'MISSING'}")
        print(f"ChromaDB: {'initialized' if health['chroma_initialized'] else 'NOT READY'}")
    except Exception as e:
        print(f"ERROR: Cannot connect to UniBot: {e}")
        return

    print("\n" + "=" * 70)
    print("Running 10 evaluation queries...")
    print("=" * 70)

    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/10] {test['category']}: {test['query'][:50]}...")
        result = run_query(test)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Time: {result['elapsed_s']}s | Confidence: {result['confidence']}")
            print(f"  Citations: {result['num_citations']} | Retrieval: {result['used_retrieval']}")
            print(f"  Language: {result['effective_language']} | Keywords: {result['keyword_match_ratio']}")
            print(f"  Answer: {result['answer_preview'][:100]}...")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results!")
        return

    # Overall metrics
    avg_time = sum(r["elapsed_s"] for r in valid) / len(valid)
    avg_confidence = sum(r["confidence"] for r in valid) / len(valid)
    retrieval_rate = sum(1 for r in valid if r["used_retrieval"]) / len(valid)
    citation_rate = sum(1 for r in valid if r["has_citations"]) / len(valid)
    avg_citations = sum(r["num_citations"] for r in valid) / len(valid)
    avg_keyword_match = sum(r["keyword_match_ratio"] for r in valid) / len(valid)
    handoff_rate = sum(1 for r in valid if r["needs_human_handoff"]) / len(valid)

    print(f"\nTotal queries:       {len(valid)}")
    print(f"Avg response time:   {avg_time:.2f}s")
    print(f"Avg confidence:      {avg_confidence:.3f}")
    print(f"Retrieval rate:      {retrieval_rate*100:.0f}%")
    print(f"Citation rate:       {citation_rate*100:.0f}%")
    print(f"Avg citations/resp:  {avg_citations:.1f}")
    print(f"Keyword match rate:  {avg_keyword_match*100:.0f}%")
    print(f"Human handoff rate:  {handoff_rate*100:.0f}%")

    # Per-category breakdown
    categories = {}
    for r in valid:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    print("\n--- Per-Category Breakdown ---")
    for cat, cat_results in categories.items():
        cat_avg_time = sum(r["elapsed_s"] for r in cat_results) / len(cat_results)
        cat_avg_conf = sum(r["confidence"] for r in cat_results) / len(cat_results)
        cat_keyword = sum(r["keyword_match_ratio"] for r in cat_results) / len(cat_results)
        cat_citations = sum(r["num_citations"] for r in cat_results) / len(cat_results)
        print(f"\n{cat}:")
        print(f"  Queries: {len(cat_results)} | Avg time: {cat_avg_time:.2f}s")
        print(f"  Avg confidence: {cat_avg_conf:.3f} | Keyword match: {cat_keyword*100:.0f}%")
        print(f"  Avg citations: {cat_citations:.1f}")

    # Multilingual specific
    ml_results = [r for r in valid if r["category"] == "Multilingual"]
    if ml_results:
        print("\n--- Multilingual Performance ---")
        for r in ml_results:
            print(f"  Language: {r['effective_language']} | Confidence: {r['confidence']}")
            print(f"  Time: {r['elapsed_s']}s | Citations: {r['num_citations']}")

    # Save full results to JSON
    output = {
        "summary": {
            "total_queries": len(valid),
            "avg_response_time_s": round(avg_time, 2),
            "avg_confidence": round(avg_confidence, 3),
            "retrieval_rate_pct": round(retrieval_rate * 100, 1),
            "citation_rate_pct": round(citation_rate * 100, 1),
            "avg_citations_per_response": round(avg_citations, 1),
            "keyword_match_rate_pct": round(avg_keyword_match * 100, 1),
            "human_handoff_rate_pct": round(handoff_rate * 100, 1),
        },
        "per_category": {
            cat: {
                "count": len(cat_results),
                "avg_time_s": round(sum(r["elapsed_s"] for r in cat_results) / len(cat_results), 2),
                "avg_confidence": round(sum(r["confidence"] for r in cat_results) / len(cat_results), 3),
                "keyword_match_pct": round(sum(r["keyword_match_ratio"] for r in cat_results) / len(cat_results) * 100, 1),
            }
            for cat, cat_results in categories.items()
        },
        "individual_results": results,
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
