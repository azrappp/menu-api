from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pulp


app = FastAPI(
    title="MILP Menu Recommendation API",
    description="Menu recommendation using TKPI/URT data and MILP optimization",
    version="1.0.0"
)


# ============================================================
# GLOBAL CONFIG
# ============================================================

FOOD_DATA_PATH = "./ready_food_data_frame.csv"

PORTION_STEP = 0.5
MAX_PORTION_PER_FOOD = 2.0

PORTION_PLAN = {
    1100: {"MP": 3, "LH": 2, "LN": 2, "S": 2, "B": 1, "SS": 0, "M": 3, "G": 1},
    1200: {"MP": 3, "LH": 2, "LN": 2, "S": 2, "B": 1, "SS": 1, "M": 3, "G": 2},
    1300: {"MP": 3, "LH": 2, "LN": 2, "S": 2, "B": 2, "SS": 1, "M": 4, "G": 2},
    1400: {"MP": 3, "LH": 3, "LN": 3, "S": 2, "B": 2, "SS": 0, "M": 3, "G": 3},
    1500: {"MP": 3, "LH": 3, "LN": 3, "S": 3, "B": 3, "SS": 0, "M": 4, "G": 3},
    1600: {"MP": 4, "LH": 3, "LN": 3, "S": 3, "B": 2, "SS": 0, "M": 4, "G": 2},
    1700: {"MP": 4, "LH": 3, "LN": 3, "S": 3, "B": 3, "SS": 1, "M": 3, "G": 2},
    1800: {"MP": 4, "LH": 3, "LN": 3, "S": 3, "B": 3, "SS": 1, "M": 4, "G": 3},
    1900: {"MP": 4, "LH": 4, "LN": 3, "S": 3, "B": 3, "SS": 1, "M": 4, "G": 3},
    2000: {"MP": 4, "LH": 3, "LN": 4, "S": 4, "B": 4, "SS": 1, "M": 4, "G": 4},
    2100: {"MP": 4, "LH": 3, "LN": 4, "S": 4, "B": 4, "SS": 1, "M": 4, "G": 4},
    2200: {"MP": 4, "LH": 3, "LN": 3, "S": 4, "B": 4, "SS": 2, "M": 5, "G": 4},
    2300: {"MP": 5, "LH": 4, "LN": 4, "S": 4, "B": 4, "SS": 0, "M": 5, "G": 4},
    2400: {"MP": 5, "LH": 4, "LN": 4, "S": 4, "B": 4, "SS": 1, "M": 5, "G": 4},
    2500: {"MP": 5.5, "LH": 4, "LN": 4, "S": 4, "B": 4, "SS": 1, "M": 5, "G": 4},
}


NUTRIENT_COLS = [
    "energy_kcal_per_portion",
    "protein_g_per_portion",
    "fat_g_per_portion",
    "carb_g_per_portion",
    "sodium_mg_per_portion",
    "potassium_mg_per_portion",
    "calcium_mg_per_portion",
    "fiber_g_per_portion",
]


SUMMARY_COLS = [
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carb_g",
    "sodium_mg",
    "fiber_g",
]


# ============================================================
# REQUEST MODEL
# ============================================================

class MenuRequest(BaseModel):
    energy_kcal: int = Field(..., example=1600)
    diet_type: str = Field(default="DM_HT_OBESITY", example="DM_HT_OBESITY")
    carb_g: float = Field(default=240)
    protein_g: float = Field(default=60)
    fat_g: float = Field(default=44)
    sodium_mg_max: float = Field(default=2000)
    fiber_g_min: float = Field(default=25)


# ============================================================
# LOAD DATA ONCE
# ============================================================

def load_food_data() -> pd.DataFrame:
    df = pd.read_csv(FOOD_DATA_PATH)

    old_result_cols = [
        "selected_portions",
        "selected_gram",
        "energy_kcal_selected_total",
        "protein_g_selected_total",
        "fat_g_selected_total",
        "carb_g_selected_total",
        "sodium_mg_selected_total",
        "fiber_g_selected_total",
        "meal_time",
        "meal_portion",
        "meal_gram",
        "meal_urt",
    ]

    df = df.drop(
        columns=[c for c in old_result_cols if c in df.columns],
        errors="ignore"
    )

    return df


FOOD_CANDIDATES = load_food_data()


# ============================================================
# CORE FUNCTIONS
# ============================================================

def filter_foods_for_disease(df: pd.DataFrame, diet_type: str) -> pd.DataFrame:
    filtered = df.copy()

    if "DM" in diet_type:
        filtered = filtered[filtered["category_code"] != "G"]

    if "HT" in diet_type:
        filtered = filtered[
            filtered["sodium_mg_per_portion"].isna()
            | (filtered["sodium_mg_per_portion"] <= 400)
        ]

    return filtered


def get_nearest_energy_level(energy_kcal: int) -> int:
    return min(PORTION_PLAN.keys(), key=lambda x: abs(x - energy_kcal))


def adjust_portion_plan_for_disease(base_plan: Dict[str, float], diet_type: str) -> Dict[str, float]:
    plan = base_plan.copy()

    if "DM" in diet_type:
        plan["G"] = 0

    if "OBESITY" in diet_type:
        plan["G"] = 0
        if "M" in plan:
            plan["M"] = min(plan["M"], 2)

    if diet_type in ["DM_OBESITY", "DM_HT", "DM_HT_OBESITY"]:
        current_sb = plan.get("S", 0) + plan.get("B", 0)
        if current_sb < 5:
            plan["S"] = plan.get("S", 0) + (5 - current_sb)

    return plan


def get_disease_constraints(request: MenuRequest) -> Dict[str, Any]:
    constraints = {
        "energy_target": request.energy_kcal,
        "carb_min_g": None,
        "carb_max_g": None,
        "fat_min_g": None,
        "fat_max_g": None,
        "sodium_max_mg": None,
        "fiber_min_g": None,
    }

    if request.diet_type == "DM_HT_OBESITY":
        constraints["carb_min_g"] = request.carb_g * 0.90
        constraints["carb_max_g"] = request.carb_g * 1.10
        constraints["fat_min_g"] = request.fat_g * 0.80
        constraints["fat_max_g"] = request.fat_g * 1.10
        constraints["fiber_min_g"] = request.fiber_g_min
        constraints["sodium_max_mg"] = request.sodium_mg_max

    return constraints


def prepare_milp_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    milp_df = df.copy().reset_index(drop=True)

    required_cols = [
        "category_code",
        "energy_kcal_per_portion",
        "protein_g_per_portion",
        "fat_g_per_portion",
        "carb_g_per_portion",
        "sodium_mg_per_portion",
        "fiber_g_per_portion",
    ]

    milp_df = milp_df.dropna(subset=required_cols).copy()

    numeric_cols = [
        "gram_per_portion",
        "energy_kcal_per_portion",
        "protein_g_per_portion",
        "fat_g_per_portion",
        "carb_g_per_portion",
        "sodium_mg_per_portion",
        "potassium_mg_per_portion",
        "calcium_mg_per_portion",
        "fiber_g_per_portion",
        "urt_qty",
    ]

    for col in numeric_cols:
        if col in milp_df.columns:
            milp_df[col] = pd.to_numeric(milp_df[col], errors="coerce").fillna(0)

    return milp_df


def solve_milp_menu(
    milp_df: pd.DataFrame,
    adjusted_plan: Dict[str, float],
    constraints: Dict[str, Any],
) -> pd.DataFrame:

    model = pulp.LpProblem("Menu_Recommendation", pulp.LpMinimize)

    x = {
        i: pulp.LpVariable(
            f"x_{i}",
            lowBound=0,
            upBound=int(MAX_PORTION_PER_FOOD / PORTION_STEP),
            cat="Integer"
        )
        for i in milp_df.index
    }

    portion = {i: x[i] * PORTION_STEP for i in milp_df.index}

    total_energy = pulp.lpSum(
        portion[i] * milp_df.loc[i, "energy_kcal_per_portion"]
        for i in milp_df.index
    )

    total_protein = pulp.lpSum(
        portion[i] * milp_df.loc[i, "protein_g_per_portion"]
        for i in milp_df.index
    )

    total_fat = pulp.lpSum(
        portion[i] * milp_df.loc[i, "fat_g_per_portion"]
        for i in milp_df.index
    )

    total_carb = pulp.lpSum(
        portion[i] * milp_df.loc[i, "carb_g_per_portion"]
        for i in milp_df.index
    )

    total_sodium = pulp.lpSum(
        portion[i] * milp_df.loc[i, "sodium_mg_per_portion"]
        for i in milp_df.index
    )

    total_fiber = pulp.lpSum(
        portion[i] * milp_df.loc[i, "fiber_g_per_portion"]
        for i in milp_df.index
    )

    # --------------------------------------------------------
    # Portion structure
    # Hard: M, G, SS
    # Soft: others
    # --------------------------------------------------------

    hard_categories = ["M", "G", "SS"]
    portion_dev = {}

    for category, target_portion in adjusted_plan.items():
        category_indices = milp_df[milp_df["category_code"] == category].index.tolist()

        if len(category_indices) == 0 and target_portion > 0:
            raise HTTPException(
                status_code=400,
                detail=f"No candidate food for required category {category}"
            )

        if len(category_indices) == 0:
            continue

        cat_total = pulp.lpSum(portion[i] for i in category_indices)

        if category in hard_categories:
            model += cat_total == target_portion, f"hard_portion_{category}"
        else:
            dev_pos = pulp.LpVariable(f"dev_{category}_pos", lowBound=0)
            dev_neg = pulp.LpVariable(f"dev_{category}_neg", lowBound=0)

            model += (
                cat_total - target_portion == dev_pos - dev_neg,
                f"soft_portion_{category}"
            )

            portion_dev[category] = dev_pos + dev_neg

    # --------------------------------------------------------
    # Nutrient constraints
    # --------------------------------------------------------

    energy_target = constraints["energy_target"]
    energy_min = energy_target * 0.90
    energy_max = energy_target * 1.10

    model += total_energy >= energy_min, "energy_min"
    model += total_energy <= energy_max, "energy_max"

    if constraints.get("carb_min_g") is not None:
        model += total_carb >= constraints["carb_min_g"], "carb_min"

    if constraints.get("carb_max_g") is not None:
        model += total_carb <= constraints["carb_max_g"], "carb_max"

    if constraints.get("fat_min_g") is not None:
        model += total_fat >= constraints["fat_min_g"], "fat_min"

    if constraints.get("fat_max_g") is not None:
        model += total_fat <= constraints["fat_max_g"], "fat_max"

    if constraints.get("sodium_max_mg") is not None:
        model += total_sodium <= constraints["sodium_max_mg"], "sodium_max"

    if constraints.get("fiber_min_g") is not None:
        model += total_fiber >= constraints["fiber_min_g"], "fiber_min"

    # --------------------------------------------------------
    # Variety constraints
    # --------------------------------------------------------

    max_x_value = int(MAX_PORTION_PER_FOOD / PORTION_STEP)

    # Max 1 fruit type per day
    fruit_indices = milp_df[milp_df["category_code"] == "B"].index.tolist()
    fruit_used = {i: pulp.LpVariable(f"fruit_used_{i}", cat="Binary") for i in fruit_indices}

    for i in fruit_indices:
        model += x[i] <= max_x_value * fruit_used[i]

    if fruit_indices:
        model += pulp.lpSum(fruit_used[i] for i in fruit_indices) <= 1, "max_one_fruit_type_per_day"

    # Max 2 LH types per day
    lh_indices = milp_df[milp_df["category_code"] == "LH"].index.tolist()
    lh_used = {i: pulp.LpVariable(f"lh_used_{i}", cat="Binary") for i in lh_indices}

    for i in lh_indices:
        model += x[i] <= max_x_value * lh_used[i]

    if lh_indices:
        model += pulp.lpSum(lh_used[i] for i in lh_indices) <= 2, "max_two_lh_types_per_day"

    # Max 1 oil type per day
    oil_indices = milp_df[milp_df["category_code"] == "M"].index.tolist()
    oil_used = {i: pulp.LpVariable(f"oil_used_{i}", cat="Binary") for i in oil_indices}

    for i in oil_indices:
        model += x[i] <= max_x_value * oil_used[i]

    if oil_indices:
        model += pulp.lpSum(oil_used[i] for i in oil_indices) <= 1, "max_one_oil_type_per_day"

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    energy_dev_pos = pulp.LpVariable("energy_dev_pos", lowBound=0)
    energy_dev_neg = pulp.LpVariable("energy_dev_neg", lowBound=0)

    model += (
        total_energy - energy_target == energy_dev_pos - energy_dev_neg,
        "energy_deviation"
    )

    portion_penalty = pulp.lpSum(
        20 * portion_dev[cat]
        for cat in portion_dev
    )

    model += (
        10 * (energy_dev_pos + energy_dev_neg)
        + 0.01 * total_sodium
        + portion_penalty
    ), "objective"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        raise HTTPException(
            status_code=422,
            detail=f"MILP failed. Solver status: {status}"
        )

    selected_items = []

    for i in milp_df.index:
        val = pulp.value(x[i])
        if val is not None and val > 0:
            row = milp_df.loc[i].copy()
            row["selected_portions"] = val * PORTION_STEP
            selected_items.append(row)

    if not selected_items:
        raise HTTPException(
            status_code=422,
            detail="MILP optimal but no food selected."
        )

    milp_menu = pd.DataFrame(selected_items)

    for col in NUTRIENT_COLS:
        if col in milp_menu.columns:
            total_col = col.replace("_per_portion", "_selected_total")
            milp_menu[total_col] = milp_menu[col] * milp_menu["selected_portions"]

    milp_menu["selected_gram"] = (
        milp_menu["gram_per_portion"] * milp_menu["selected_portions"]
    )

    return milp_menu


# ============================================================
# MEAL ALLOCATION
# ============================================================

def split_preserve_total(total_portion: float, ratios: Dict[str, float], step: float) -> Dict[str, float]:
    meal_names = list(ratios.keys())

    raw = {
        meal: total_portion * ratio
        for meal, ratio in ratios.items()
    }

    rounded = {
        meal: np.floor(value / step) * step
        for meal, value in raw.items()
    }

    remaining = round(total_portion - sum(rounded.values()), 10)

    remainders = sorted(
        meal_names,
        key=lambda meal: raw[meal] - rounded[meal],
        reverse=True
    )

    idx = 0
    while remaining >= step - 1e-9:
        meal = remainders[idx % len(remainders)]
        rounded[meal] += step
        remaining = round(remaining - step, 10)
        idx += 1

    return rounded


def assign_lh_items(lh_df: pd.DataFrame) -> List[pd.Series]:
    meal_rows = []
    meal_cycle = ["lunch", "dinner"]
    meal_index = 0

    for _, row in lh_df.iterrows():
        remaining = row["selected_portions"]

        while remaining > 0:
            portion_to_assign = min(remaining, 1.0)
            meal_name = meal_cycle[meal_index % len(meal_cycle)]

            meal_row = row.copy()
            meal_row["meal_time"] = meal_name
            meal_row["meal_portion"] = portion_to_assign
            meal_row["meal_gram"] = portion_to_assign * row["gram_per_portion"]
            meal_row["meal_urt_qty"] = portion_to_assign * row["urt_qty"]
            meal_row["meal_urt"] = f"{meal_row['meal_urt_qty']} {row['urt_unit']}"

            for col in NUTRIENT_COLS:
                if col in row.index:
                    meal_col = col.replace("_per_portion", "")
                    meal_row[meal_col] = portion_to_assign * row[col]

            meal_rows.append(meal_row)

            remaining -= portion_to_assign
            meal_index += 1

    return meal_rows


def allocate_meals(milp_menu: pd.DataFrame) -> pd.DataFrame:
    meal_rules = {
        "MP": {
            "breakfast": 0.4,
            "lunch": 0.3,
            "dinner": 0.3,
        },
        "LN": {
            "breakfast": 0.3,
            "lunch": 0.3,
            "dinner": 0.4,
        },
        "S": {
            "lunch": 0.5,
            "dinner": 0.5,
        },
        "B": {
            "morning_snack": 0.5,
            "afternoon_snack": 0.5,
        },
        "M": {
            "lunch": 0.5,
            "dinner": 0.5,
        },
    }

    step_size = {
        "MP": 0.5,
        "LH": 0.5,
        "LN": 0.5,
        "S": 0.5,
        "B": 0.5,
        "M": 0.5,
    }

    meal_rows = []

    lh_df = milp_menu[milp_menu["category_code"] == "LH"].copy()
    meal_rows.extend(assign_lh_items(lh_df))

    other_menu = milp_menu[milp_menu["category_code"] != "LH"].copy()

    for _, row in other_menu.iterrows():
        category = row["category_code"]

        if category not in meal_rules:
            continue

        rules = meal_rules[category]
        step = step_size[category]

        meal_portions = split_preserve_total(
            total_portion=row["selected_portions"],
            ratios=rules,
            step=step,
        )

        for meal_name, rounded_portion in meal_portions.items():
            if rounded_portion <= 0:
                continue

            meal_row = row.copy()
            meal_row["meal_time"] = meal_name
            meal_row["meal_portion"] = rounded_portion
            meal_row["meal_gram"] = rounded_portion * row["gram_per_portion"]
            meal_row["meal_urt_qty"] = rounded_portion * row["urt_qty"]
            meal_row["meal_urt"] = f"{meal_row['meal_urt_qty']} {row['urt_unit']}"

            for col in NUTRIENT_COLS:
                if col in row.index:
                    meal_col = col.replace("_per_portion", "")
                    meal_row[meal_col] = rounded_portion * row[col]

            meal_rows.append(meal_row)

    return pd.DataFrame(meal_rows)


# ============================================================
# RESPONSE FORMATTER
# ============================================================

def format_response(
    request: MenuRequest,
    milp_menu: pd.DataFrame,
    meal_df: pd.DataFrame,
    adjusted_plan: Dict[str, float],
) -> Dict[str, Any]:

    daily_total = meal_df[SUMMARY_COLS].sum().round(2).to_dict()

    meal_order = [
        "breakfast",
        "morning_snack",
        "lunch",
        "afternoon_snack",
        "dinner",
    ]

    meals = []

    for meal_time in meal_order:
        meal_items_df = meal_df[meal_df["meal_time"] == meal_time]

        if meal_items_df.empty:
            continue

        items = []

        for _, row in meal_items_df.iterrows():
            items.append({
                "food_name": row["food_name"],
                "category_code": row["category_code"],
                "portion": float(row["meal_portion"]),
                "urt": row["meal_urt"],
                "gram": round(float(row["meal_gram"]), 2),
                "energy_kcal": round(float(row["energy_kcal"]), 2),
                "protein_g": round(float(row["protein_g"]), 2),
                "fat_g": round(float(row["fat_g"]), 2),
                "carb_g": round(float(row["carb_g"]), 2),
                "fiber_g": round(float(row["fiber_g"]), 2),
            })

        meal_summary = meal_items_df[SUMMARY_COLS].sum().round(2).to_dict()

        meals.append({
            "meal_time": meal_time,
            "summary": meal_summary,
            "items": items,
        })

    return {
        "status": "success",
        "message": "Menu recommendation generated successfully.",
        "input": {
            "diet_type": request.diet_type,
            "energy_kcal": request.energy_kcal,
            "carb_g": request.carb_g,
            "protein_g": request.protein_g,
            "fat_g": request.fat_g,
            "sodium_mg_max": request.sodium_mg_max,
            "fiber_g_min": request.fiber_g_min,
        },
        "adjusted_portion_plan": adjusted_plan,
        "daily_total": daily_total,
        "meals": meals,
    }


# ============================================================
# API ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "MILP Menu Recommendation API is running."
    }


@app.post("/recommend-menu")
def recommend_menu(request: MenuRequest):
    candidate_for_patient = filter_foods_for_disease(
        FOOD_CANDIDATES,
        request.diet_type,
    )

    milp_df = prepare_milp_dataframe(candidate_for_patient)

    nearest_energy = get_nearest_energy_level(request.energy_kcal)
    base_plan = PORTION_PLAN[nearest_energy]
    adjusted_plan = adjust_portion_plan_for_disease(
        base_plan,
        request.diet_type,
    )

    constraints = get_disease_constraints(request)

    milp_menu = solve_milp_menu(
        milp_df=milp_df,
        adjusted_plan=adjusted_plan,
        constraints=constraints,
    )

    meal_df = allocate_meals(milp_menu)

    return format_response(
        request=request,
        milp_menu=milp_menu,
        meal_df=meal_df,
        adjusted_plan=adjusted_plan,
    )