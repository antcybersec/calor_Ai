"""
Unit & Integration Tests for CalorAI Database, Tools, and Agent Logic.
"""

import os
import pytest
from database import (
    init_db,
    log_meal,
    correct_last_meal,
    get_daily_totals,
    save_memory,
    get_memories,
    clear_user_data
)
from nutrition import get_nutrition_info

TEST_DB = "test_calor_ai.db"
TEST_USER = "unittest_user"

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    init_db(TEST_DB)
    clear_user_data(TEST_USER, db_path=TEST_DB)
    yield
    clear_user_data(TEST_USER, db_path=TEST_DB)
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass

def test_log_meal_and_daily_totals():
    meal = log_meal(
        user_id=TEST_USER,
        raw_input="2 rotis and dal",
        items=[
            {"name": "Roti", "calories": 200, "protein_g": 6.4},
            {"name": "Dal", "calories": 150, "protein_g": 9.0}
        ],
        total_calories=350.0,
        total_protein_g=15.4,
        db_path=TEST_DB
    )
    assert meal["meal_id"] is not None
    assert meal["total_calories"] == 350.0

    totals = get_daily_totals(TEST_USER, db_path=TEST_DB)
    assert totals["meal_count"] == 1
    assert totals["total_calories"] == 350.0
    assert totals["total_protein_g"] == 15.4

def test_meal_correction_no_double_counting():
    # 1. Log initial meal: 2 rotis (200 kcal)
    log_meal(
        user_id=TEST_USER,
        raw_input="2 rotis",
        items=[{"name": "Roti", "quantity": 2, "calories": 200}],
        total_calories=200.0,
        total_protein_g=6.4,
        db_path=TEST_DB
    )
    t1 = get_daily_totals(TEST_USER, db_path=TEST_DB)
    assert t1["total_calories"] == 200.0

    # 2. User corrects: "actually that was 3 rotis not 2" (300 kcal)
    correct_last_meal(
        user_id=TEST_USER,
        raw_input="actually that was 3 rotis not 2",
        items=[{"name": "Roti", "quantity": 3, "calories": 300}],
        total_calories=300.0,
        total_protein_g=9.6,
        db_path=TEST_DB
    )

    # 3. Verify totals update to 300 kcal (NOT 500 kcal!)
    t2 = get_daily_totals(TEST_USER, db_path=TEST_DB)
    assert t2["total_calories"] == 300.0
    assert t2["total_protein_g"] == 9.6
    assert t2["meal_count"] == 1  # Only 1 active meal

def test_persistent_memory_storage():
    save_memory(
        user_id=TEST_USER,
        category="preference",
        memory_key="dietary_preference",
        memory_value="Vegetarian",
        db_path=TEST_DB
    )
    mems = get_memories(TEST_USER, db_path=TEST_DB)
    assert len(mems) == 1
    assert mems[0]["memory_value"] == "Vegetarian"

def test_nutrition_lookup():
    paratha_info = get_nutrition_info("paratha", quantity=2)
    assert paratha_info is not None
    assert paratha_info["calories"] == 520.0  # 260 * 2
    assert paratha_info["protein_g"] == 12.0
