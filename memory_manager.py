"""
Selective Memory Extraction and Retrieval System.
Decouples persistent facts (dietary rules, usual meals, macro targets) from conversation history.
Stores facts in SQLite database and selectively injects them into system prompt context.
Works with both OpenAI and Google Gemini as the extraction LLM.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from database import get_memories, save_memory, get_meal_history, get_daily_totals

load_dotenv()

def _get_extraction_llm():
    """Return an LLM instance for memory extraction, preferring Gemini if available."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key and not gemini_key.startswith("your_"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_key,
            temperature=0.0
        )
    if openai_key and not openai_key.startswith("your_") and openai_key != "mock_key":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.0)
    return None


def extract_and_save_memories(user_id: str, user_text: str, assistant_response: str = "") -> List[Dict[str, Any]]:
    """
    Analyzes user text to extract long-term persistent memories (dietary preferences,
    'my usual' meal definitions, macro targets, eating habits).
    Stores extracted memories in SQLite database.
    Uses both fast regex rules AND LLM-assisted extraction (Gemini or OpenAI).
    """
    text_lower = user_text.lower().strip()
    extracted = []

    # --- Fast Regex Rule Matching ---
    if "vegetarian" in text_lower or "veg btw" in text_lower or "i am veg" in text_lower or "i'm veg" in text_lower:
        m = save_memory(user_id, "preference", "dietary_preference", "Vegetarian")
        extracted.append(m)
    elif "vegan" in text_lower:
        m = save_memory(user_id, "preference", "dietary_preference", "Vegan")
        extracted.append(m)
    elif "eggetarian" in text_lower or "eggitarian" in text_lower:
        m = save_memory(user_id, "preference", "dietary_preference", "Eggetarian")
        extracted.append(m)

    # Protein targets
    if "protein" in text_lower and any(w in text_lower for w in ["target", "goal", "aiming", "targeting", "need"]):
        match = re.search(r'(\d+)\s*g', text_lower)
        if match:
            target_val = f"{match.group(1)}g protein daily target"
            m = save_memory(user_id, "goal", "protein_target", target_val)
            extracted.append(m)

    # Calorie targets
    if "calorie" in text_lower and any(w in text_lower for w in ["target", "goal", "aiming", "limit"]):
        match = re.search(r'(\d+)\s*(?:kcal|cal|calories)', text_lower)
        if match:
            m = save_memory(user_id, "goal", "calorie_target", f"{match.group(1)} kcal daily target")
            extracted.append(m)

    # "My usual" definitions
    if "my usual" in text_lower and ("is" in text_lower or "means" in text_lower or "="):
        # Try to parse what comes after "my usual is"
        match = re.search(r"my usual (?:is|means|=)\s*(.+)", text_lower)
        if match:
            usual_val = match.group(1).strip()
            m = save_memory(user_id, "usual_meal", "my_usual", usual_val)
            extracted.append(m)

    # --- LLM-Assisted Extraction for Complex Facts ---
    # Only trigger for messages that likely contain memorable facts
    trigger_words = ["usual", "favorite", "always", "normally", "typically", "every day",
                     "goal", "target", "plan", "trying to", "never eat", "don't eat", "hate"]
    should_extract_with_llm = any(w in text_lower for w in trigger_words)

    if should_extract_with_llm:
        llm = _get_extraction_llm()
        if llm:
            try:
                prompt = (
                    "You are a memory extractor for a meal tracking assistant.\n"
                    f"User message: '{user_text}'\n\n"
                    "Extract ONLY persistent long-term user facts worth remembering across sessions:\n"
                    "- Dietary restrictions (vegetarian, gluten-free, allergies)\n"
                    "- Named meal aliases ('my usual' = specific meal description)\n"
                    "- Nutritional goals (protein, calorie targets)\n"
                    "- Consistent eating habits (skips lunch, grazes in evenings)\n\n"
                    "IMPORTANT: Only extract facts explicitly stated, not inferred.\n"
                    "Return ONLY a JSON array (or [] if nothing to extract):\n"
                    '[{"category": "preference|usual_meal|goal|habit", "key": "short_key", "value": "stored value"}]'
                )
                res = llm.invoke(prompt)
                clean_res = res.content.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_res)

                for item in parsed:
                    cat = item.get("category", "general")
                    key = item.get("key", "info").lower().replace(" ", "_")
                    val = item.get("value", "")
                    if val and len(val) > 1:
                        # Don't overwrite things we already caught with regex
                        already_saved = any(e.get("memory_key") == key for e in extracted)
                        if not already_saved:
                            m = save_memory(user_id, cat, key, val)
                            extracted.append(m)
            except Exception as e:
                pass  # Silent — memory extraction failure should never break a conversation turn

    return extracted


def format_memories_for_prompt(user_id: str) -> str:
    """
    Formats all persistent user memories into a compact, prompt-friendly context section.
    Injects only the most relevant facts without bloating the prompt.
    """
    memories = get_memories(user_id)
    if not memories:
        return ""

    lines = ["[USER PROFILE & PERSISTENT MEMORY]"]
    for m in memories:
        cat = m["category"].title()
        key = m["memory_key"].replace("_", " ").title()
        val = m["memory_value"]
        lines.append(f"• {key}: {val}")

    return "\n".join(lines)


def resolve_meal_reference(user_id: str, text: str) -> Optional[Dict[str, Any]]:
    """
    Resolves phrases like 'same as yesterday' or 'my usual' to explicit meal items
    with accurate nutritional data from stored history/memories.
    """
    text_clean = text.lower().strip()

    # Case A: "same as yesterday" / "like yesterday"
    if "same as yesterday" in text_clean or "like yesterday" in text_clean or "yesterday's" in text_clean:
        from datetime import date, timedelta
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        yesterday_meals = get_meal_history(user_id, limit=10, query_date=yesterday_str)

        if yesterday_meals:
            # Sum all of yesterday's meals if user says "same as yesterday" broadly
            total_cals = sum(m["total_calories"] for m in yesterday_meals)
            total_prot = sum(m["total_protein_g"] for m in yesterday_meals)
            total_carbs = sum(m["total_carbs_g"] for m in yesterday_meals)
            total_fat = sum(m["total_fat_g"] for m in yesterday_meals)
            all_items = []
            for m in yesterday_meals:
                all_items.extend(m["items"])

            return {
                "source": "yesterday_full_day",
                "date": yesterday_str,
                "raw_input": f"same as yesterday ({len(yesterday_meals)} meals)",
                "items": all_items,
                "total_calories": round(total_cals, 1),
                "total_protein_g": round(total_prot, 1),
                "total_carbs_g": round(total_carbs, 1),
                "total_fat_g": round(total_fat, 1)
            }
        else:
            # Fallback to most recent meal ever logged
            all_recent = get_meal_history(user_id, limit=1)
            if all_recent:
                m = all_recent[0]
                return {
                    "source": "last_logged_meal_fallback",
                    "raw_input": m["raw_input"],
                    "items": m["items"],
                    "total_calories": m["total_calories"],
                    "total_protein_g": m["total_protein_g"],
                    "total_carbs_g": m["total_carbs_g"],
                    "total_fat_g": m["total_fat_g"]
                }
        return None

    # Case B: "my usual" / "the usual"
    if "my usual" in text_clean or "the usual" in text_clean:
        memories = get_memories(user_id)

        # Look for an explicit "my_usual" saved memory key
        usual_mem = next(
            (m for m in memories if m["memory_key"] == "my_usual" or m["category"] == "usual_meal"),
            None
        )

        if usual_mem:
            # We have a defined usual meal — look up its nutrition from the meal history
            usual_desc = usual_mem["memory_value"]

            # Try to find a past logged meal matching this description
            history = get_meal_history(user_id, limit=20)
            for past_meal in history:
                if any(word in past_meal["raw_input"].lower() for word in usual_desc.lower().split()):
                    return {
                        "source": "usual_memory_matched_history",
                        "description": usual_desc,
                        "items": past_meal["items"],
                        "total_calories": past_meal["total_calories"],
                        "total_protein_g": past_meal["total_protein_g"],
                        "total_carbs_g": past_meal["total_carbs_g"],
                        "total_fat_g": past_meal["total_fat_g"]
                    }

            # No history match — return the memory description with a note for LLM to estimate
            return {
                "source": "usual_memory_needs_estimation",
                "description": usual_desc,
                "items": [{"name": usual_desc, "portion": "1 serving", "needs_estimation": True}],
                "total_calories": None,  # Signal to LLM to estimate
                "total_protein_g": None,
                "total_carbs_g": None,
                "total_fat_g": None
            }
        else:
            # No explicit usual defined — use most common/recent breakfast
            history = get_meal_history(user_id, limit=10)
            breakfast_meals = [m for m in history if m.get("meal_type") in ("breakfast", "unknown")]
            if breakfast_meals:
                most_recent = breakfast_meals[0]
                return {
                    "source": "inferred_usual_from_history",
                    "description": most_recent["raw_input"],
                    "items": most_recent["items"],
                    "total_calories": most_recent["total_calories"],
                    "total_protein_g": most_recent["total_protein_g"],
                    "total_carbs_g": most_recent["total_carbs_g"],
                    "total_fat_g": most_recent["total_fat_g"]
                }

    return None
