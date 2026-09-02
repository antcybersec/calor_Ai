"""
Nutritional Lookup Table & Estimation Engine.
Contains accurate values for common meal items (Indian & International),
providing low-latency, deterministic nutrition lookup with macro breakdowns.
"""

from typing import Dict, Any, Optional

NUTRITION_DATABASE: Dict[str, Dict[str, float]] = {
    # Indian Standard Breakfast & Mains
    "paratha": {"calories": 260.0, "protein_g": 6.0, "carbs_g": 36.0, "fat_g": 11.0, "serving": "1 paratha"},
    "aloo paratha": {"calories": 290.0, "protein_g": 6.5, "carbs_g": 42.0, "fat_g": 11.5, "serving": "1 paratha"},
    "roti": {"calories": 100.0, "protein_g": 3.2, "carbs_g": 18.0, "fat_g": 1.5, "serving": "1 roti"},
    "chapati": {"calories": 100.0, "protein_g": 3.2, "carbs_g": 18.0, "fat_g": 1.5, "serving": "1 chapati"},
    "chai": {"calories": 80.0, "protein_g": 2.0, "carbs_g": 10.0, "fat_g": 3.0, "serving": "1 cup"},
    "tea": {"calories": 80.0, "protein_g": 2.0, "carbs_g": 10.0, "fat_g": 3.0, "serving": "1 cup"},
    "masala chai": {"calories": 90.0, "protein_g": 2.5, "carbs_g": 12.0, "fat_g": 3.5, "serving": "1 cup"},
    "biryani": {"calories": 450.0, "protein_g": 22.0, "carbs_g": 52.0, "fat_g": 16.0, "serving": "1 portion"},
    "chicken biryani": {"calories": 480.0, "protein_g": 28.0, "carbs_g": 50.0, "fat_g": 17.0, "serving": "1 portion"},
    "veg biryani": {"calories": 380.0, "protein_g": 10.0, "carbs_g": 58.0, "fat_g": 12.0, "serving": "1 portion"},
    "dal": {"calories": 150.0, "protein_g": 9.0, "carbs_g": 22.0, "fat_g": 3.0, "serving": "1 bowl"},
    "dal makhani": {"calories": 260.0, "protein_g": 10.0, "carbs_g": 24.0, "fat_g": 14.0, "serving": "1 bowl"},
    "rice": {"calories": 180.0, "protein_g": 3.5, "carbs_g": 40.0, "fat_g": 0.5, "serving": "1 cup cooked"},
    "steamed rice": {"calories": 180.0, "protein_g": 3.5, "carbs_g": 40.0, "fat_g": 0.5, "serving": "1 cup cooked"},
    "paneer tikka": {"calories": 320.0, "protein_g": 18.0, "carbs_g": 8.0, "fat_g": 24.0, "serving": "1 plate"},
    "paneer butter masala": {"calories": 380.0, "protein_g": 14.0, "carbs_g": 14.0, "fat_g": 30.0, "serving": "1 bowl"},
    "dosa": {"calories": 220.0, "protein_g": 4.5, "carbs_g": 35.0, "fat_g": 7.0, "serving": "1 dosa"},
    "masala dosa": {"calories": 330.0, "protein_g": 6.5, "carbs_g": 48.0, "fat_g": 12.0, "serving": "1 dosa"},
    "idli": {"calories": 65.0, "protein_g": 2.0, "carbs_g": 13.0, "fat_g": 0.3, "serving": "1 piece"},
    "sambar": {"calories": 110.0, "protein_g": 4.5, "carbs_g": 16.0, "fat_g": 3.0, "serving": "1 bowl"},
    "poha": {"calories": 250.0, "protein_g": 4.0, "carbs_g": 42.0, "fat_g": 7.0, "serving": "1 plate"},
    "upma": {"calories": 210.0, "protein_g": 4.5, "carbs_g": 34.0, "fat_g": 6.0, "serving": "1 bowl"},
    
    # Common Proteins & Breakfast Items
    "egg": {"calories": 70.0, "protein_g": 6.0, "carbs_g": 0.5, "fat_g": 5.0, "serving": "1 egg"},
    "boiled egg": {"calories": 70.0, "protein_g": 6.0, "carbs_g": 0.5, "fat_g": 5.0, "serving": "1 egg"},
    "omelette": {"calories": 180.0, "protein_g": 12.0, "carbs_g": 2.0, "fat_g": 14.0, "serving": "2 egg omelette"},
    "chicken breast": {"calories": 165.0, "protein_g": 31.0, "carbs_g": 0.0, "fat_g": 3.6, "serving": "100g cooked"},
    "grilled chicken": {"calories": 220.0, "protein_g": 30.0, "carbs_g": 1.0, "fat_g": 9.0, "serving": "150g"},
    "oats": {"calories": 150.0, "protein_g": 5.0, "carbs_g": 27.0, "fat_g": 2.5, "serving": "1 bowl"},
    "oatmeal": {"calories": 150.0, "protein_g": 5.0, "carbs_g": 27.0, "fat_g": 2.5, "serving": "1 bowl"},
    "milk": {"calories": 120.0, "protein_g": 6.0, "carbs_g": 9.0, "fat_g": 6.0, "serving": "1 glass (200ml)"},
    "protein shake": {"calories": 160.0, "protein_g": 25.0, "carbs_g": 4.0, "fat_g": 2.5, "serving": "1 scoop with water"},
    "whey protein": {"calories": 120.0, "protein_g": 24.0, "carbs_g": 2.0, "fat_g": 1.5, "serving": "1 scoop"},
    "bread": {"calories": 80.0, "protein_g": 3.0, "carbs_g": 14.0, "fat_g": 1.0, "serving": "1 slice"},
    "toast": {"calories": 80.0, "protein_g": 3.0, "carbs_g": 14.0, "fat_g": 1.0, "serving": "1 slice"},
    "apple": {"calories": 95.0, "protein_g": 0.5, "carbs_g": 25.0, "fat_g": 0.3, "serving": "1 medium apple"},
    "banana": {"calories": 105.0, "protein_g": 1.3, "carbs_g": 27.0, "fat_g": 0.4, "serving": "1 medium banana"},
}

def get_nutrition_info(query: str, quantity: float = 1.0) -> Optional[Dict[str, Any]]:
    """Lookup nutrition details for a food item."""
    query_clean = query.lower().strip()
    
    # Direct match or substring match
    matched_key = None
    if query_clean in NUTRITION_DATABASE:
        matched_key = query_clean
    else:
        for key in NUTRITION_DATABASE:
            if key in query_clean or query_clean in key:
                matched_key = key
                break

    if matched_key:
        base = NUTRITION_DATABASE[matched_key]
        return {
            "name": matched_key.title(),
            "quantity": quantity,
            "serving": base["serving"],
            "calories": round(base["calories"] * quantity, 1),
            "protein_g": round(base["protein_g"] * quantity, 1),
            "carbs_g": round(base["carbs_g"] * quantity, 1),
            "fat_g": round(base["fat_g"] * quantity, 1),
            "source": "database_exact"
        }
    
    return None
