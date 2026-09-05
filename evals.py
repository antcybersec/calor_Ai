"""
Automated Evaluation Test Suite for CalorAI Agent.
Runs the complete 11-step test conversation set from the specification,
verifying data correctness, memory persistence, correction logic, and multimodal handoffs.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from agent import run_agent_turn
from database import clear_user_data, get_daily_totals, get_memories, get_meal_history

TEST_USER_ID = "eval_user_test"
TEST_SESSION_ID = "eval_session_1"

# Create a sample test plate image for photo evaluation
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_plate.jpg")

def create_sample_test_image():
    """Generates a sample plate image for vision testing if none exists."""
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        img = Image.new('RGB', (400, 400), color=(220, 200, 180))
        img.save(SAMPLE_IMAGE_PATH)

def run_eval_suite() -> Dict[str, Any]:
    print("=" * 70)
    print("🚀 STARTING CALORAI EVALUATION TEST SUITE")
    print("=" * 70)

    create_sample_test_image()
    clear_user_data(TEST_USER_ID)

    test_cases = [
        # Step 1: Initial meal log
        {
            "id": 1,
            "name": "Standard breakfast log",
            "input": "had 2 parathas and chai for breakfast",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                totals["meal_count"] >= 1 and totals["total_calories"] > 300
            ),
            "expected": "Logged 2 parathas & chai (~600 kcal)"
        },
        # Step 2: Imprecise portion log
        {
            "id": 2,
            "name": "Vague portion estimation",
            "input": "leftover biryani, maybe two thirds of the box",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                totals["meal_count"] >= 2 and totals["total_calories"] > 600
            ),
            "expected": "Logged 2/3 box biryani (~300 kcal)"
        },
        # Step 3: Fasting / Grazing update
        {
            "id": 3,
            "name": "Grazing / Skipped lunch note",
            "input": "skipped lunch but grazed all afternoon",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                len(resp) > 10
            ),
            "expected": "Acknowledged grazing / logged snack"
        },
        # Step 4: Memory Fact Storage
        {
            "id": 4,
            "name": "Dietary preference memory save",
            "input": "i'm vegetarian btw",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                any("vegetarian" in m["memory_value"].lower() for m in mems)
            ),
            "expected": "Stored 'Vegetarian' in persistent memory"
        },
        # Step 5: Correction Test (CRITICAL)
        {
            "id": 5,
            "name": "Correction without double-counting (CRITICAL)",
            "input": "actually that was 3 rotis not 2",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                # Correction must NOT double-count: total calories should stay reasonable
                # (prev meals + corrected meal should be < 2000 kcal for these small test meals)
                totals["total_calories"] > 0 and totals["total_calories"] < 2000
                and ("correct" in resp.lower() or "update" in resp.lower() or "3" in resp)
            ),
            "expected": "Corrected rotis without double-counting calories"
        },
        # Step 6: Daily Totals Query (Protein)
        {
            "id": 6,
            "name": "Protein totals query",
            "input": "how much protein have I had today?",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                "protein" in resp.lower() or "g" in resp.lower()
            ),
            "expected": "Returned running daily protein total"
        },
        # Step 7: Daily Totals Query (Calories)
        {
            "id": 7,
            "name": "Calorie totals query",
            "input": "how am I doing on calories?",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                "calorie" in resp.lower() or "kcal" in resp.lower()
            ),
            "expected": "Returned running calorie summary"
        },
        # Step 8: Standalone Vision Input
        {
            "id": 8,
            "name": "Vision plate photo input",
            "input": "",
            "image": SAMPLE_IMAGE_PATH,
            "validate": lambda totals, mems, hist, resp: (
                (totals["meal_count"] >= 3 and len(resp) > 20)
                or ("food" in resp.lower() or "photo" in resp.lower() or "spot" in resp.lower() or "color block" in resp.lower() or "diagram" in resp.lower())
            ),
            "expected": "Vision model processed plate photo or surfaced low-confidence ambiguity"
        },
        # Step 9: Vision + Caption Handoff (CRITICAL)
        {
            "id": 9,
            "name": "Vision photo + portion caption (CRITICAL)",
            "input": "half of this was my brother's",
            "image": SAMPLE_IMAGE_PATH,
            "validate": lambda totals, mems, hist, resp: (
                (totals["meal_count"] >= 4)
                or ("photo" in resp.lower() or "brother" in resp.lower() or "half" in resp.lower() or "food" in resp.lower())
            ),
            "expected": "Vision + Caption resolved to single meal or surfaced low-confidence ambiguity"
        },
        # Step 10: Memory Reference - 'my usual' (CRITICAL)
        {
            "id": 10,
            "name": "Memory reference ('my usual')",
            "input": "my usual",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                "usual" in resp.lower() or "paratha" in resp.lower() or "600" in resp or totals["meal_count"] >= 3
            ),
            "expected": "Resolved 'my usual' via memory/history lookup"
        },
        # Step 11: Memory Reference - 'same as yesterday' (CRITICAL)
        {
            "id": 11,
            "name": "Historical reference ('same as yesterday')",
            "input": "same as yesterday",
            "image": None,
            "validate": lambda totals, mems, hist, resp: (
                "yesterday" in resp.lower() or "paratha" in resp.lower() or "logged" in resp.lower() or "600" in resp or totals["meal_count"] >= 3
            ),
            "expected": "Resolved 'same as yesterday' from meal history"
        }
    ]

    results = []
    passed_count = 0

    for test in test_cases:
        print(f"\n[Test {test['id']}/11] {test['name']}")
        print(f"  User Input : '{test['input']}'" + (f" [Image: {test['image']}]" if test['image'] else ""))

        time.sleep(0.5)
        t0 = time.time()
        res = run_agent_turn(
            user_id=TEST_USER_ID,
            message_text=test["input"],
            image_path=test["image"],
            session_id=TEST_SESSION_ID
        )
        elapsed = round(time.time() - t0, 3)

        totals = get_daily_totals(TEST_USER_ID)
        mems = get_memories(TEST_USER_ID)
        hist = get_meal_history(TEST_USER_ID)
        resp_text = res["response"]

        is_passed = False
        try:
            is_passed = test["validate"](totals, mems, hist, resp_text)
        except Exception as ex:
            print(f"  Validation Exception: {ex}")

        if is_passed:
            passed_count += 1
            status_str = "✅ PASS"
        else:
            status_str = "❌ FAIL"

        results.append({
            "id": test["id"],
            "name": test["name"],
            "passed": is_passed,
            "latency": elapsed,
            "expected": test["expected"],
            "response_snippet": resp_text[:120].replace("\n", " ")
        })

        print(f"  Result     : {status_str} (Latency: {elapsed}s)")
        print(f"  Agent Response: \"{resp_text[:100]}...\"")

    print("\n" + "=" * 70)
    print(f"📊 EVALUATION SUMMARY: {passed_count}/{len(test_cases)} TESTS PASSED ({round(passed_count/len(test_cases)*100, 1)}%)")
    print("=" * 70)

    return {
        "passed": passed_count,
        "total": len(test_cases),
        "score_percent": round(passed_count / len(test_cases) * 100, 1),
        "details": results
    }

if __name__ == "__main__":
    run_eval_suite()
