"""
LangChain / LangGraph Agent Tools.
Provides clean tool surfaces for meal logging, meal corrections, daily totals,
nutrition database lookups, and persistent memory management.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from database import (
    log_meal,
    correct_last_meal,
    get_daily_totals,
    get_meal_history,
    save_memory,
    get_memories,
)
from nutrition import get_nutrition_info

@tool
def log_meal_tool(
    user_id: str,
    raw_input: str,
    items_json: str,
    total_calories: float,
    total_protein_g: float = 0.0,
    total_carbs_g: float = 0.0,
    total_fat_g: float = 0.0,
    meal_type: str = "unknown"
) -> str:
    """
    Log a newly consumed meal for the user.
    `items_json` must be a JSON string array of items: [{"name": "item", "portion": "2 rotis", "calories": 200, "protein_g": 6, "carbs_g": 36, "fat_g": 3}]
    """
    try:
        items = json.loads(items_json) if isinstance(items_json, str) else items_json
    except Exception:
        items = [{"name": raw_input, "calories": total_calories}]

    res = log_meal(
        user_id=user_id,
        raw_input=raw_input,
        items=items,
        total_calories=total_calories,
        total_protein_g=total_protein_g,
        total_carbs_g=total_carbs_g,
        total_fat_g=total_fat_g,
        meal_type=meal_type
    )
    return json.dumps({
        "status": "success",
        "message": f"Successfully logged meal: {raw_input} ({total_calories} kcal)",
        "logged_meal": res
    })

@tool
def correct_meal_tool(
    user_id: str,
    raw_input: str,
    items_json: str,
    total_calories: float,
    total_protein_g: float = 0.0,
    total_carbs_g: float = 0.0,
    total_fat_g: float = 0.0
) -> str:
    """
    Correct or update the most recent logged meal (e.g., when user says 'actually that was 3 rotis not 2').
    Replaces the previous entry to prevent double-counting.
    `items_json` must be a JSON string array of corrected items.
    """
    try:
        items = json.loads(items_json) if isinstance(items_json, str) else items_json
    except Exception:
        items = [{"name": raw_input, "calories": total_calories}]

    res = correct_last_meal(
        user_id=user_id,
        raw_input=raw_input,
        items=items,
        total_calories=total_calories,
        total_protein_g=total_protein_g,
        total_carbs_g=total_carbs_g,
        total_fat_g=total_fat_g
    )
    return json.dumps({
        "status": "success",
        "message": f"Successfully updated previous meal with corrected values: {total_calories} kcal (no double counting).",
        "corrected_meal": res
    })

@tool
def get_daily_totals_tool(user_id: str, query_date: str = "") -> str:
    """
    Retrieve running daily totals (calories, protein, carbs, fat) and list of logged active meals for today (or query_date: YYYY-MM-DD).
    """
    totals = get_daily_totals(user_id=user_id, query_date=query_date if query_date else None)
    return json.dumps(totals)

@tool
def get_meal_history_tool(user_id: str, limit: int = 5) -> str:
    """
    Fetch recent active logged meals for user context or reference.
    """
    history = get_meal_history(user_id=user_id, limit=limit)
    return json.dumps(history)

@tool
def lookup_nutrition_tool(food_item: str, quantity: float = 1.0) -> str:
    """
    Lookup nutritional information (calories, protein, carbs, fat) for a specific food item and quantity.
    """
    info = get_nutrition_info(query=food_item, quantity=quantity)
    if info:
        return json.dumps(info)
    return json.dumps({
        "name": food_item,
        "found": False,
        "message": f"Exact item '{food_item}' not in standard DB; estimate based on general knowledge."
    })

@tool
def save_memory_tool(user_id: str, category: str, memory_key: str, memory_value: str) -> str:
    """
    Store long-term user fact or preference (category: preference, usual_meal, goal, habit).
    Example: save_memory_tool(user_id, "preference", "dietary", "Vegetarian")
    Example: save_memory_tool(user_id, "usual_meal", "breakfast", "2 parathas and chai")
    """
    res = save_memory(user_id=user_id, category=category, memory_key=memory_key, memory_value=memory_value)
    return json.dumps({
        "status": "success",
        "message": f"Saved long-term memory fact: [{category}] {memory_key} = {memory_value}",
        "memory": res
    })

@tool
def get_memories_tool(user_id: str) -> str:
    """
    Retrieve all long-term persistent memories stored for user.
    """
    memories = get_memories(user_id=user_id)
    return json.dumps(memories)

ALL_TOOLS = [
    log_meal_tool,
    correct_meal_tool,
    get_daily_totals_tool,
    get_meal_history_tool,
    lookup_nutrition_tool,
    save_memory_tool,
    get_memories_tool
]
