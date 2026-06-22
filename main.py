from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pulp

app = FastAPI(
    title="MILP Menu Recommendation API",
    description="Menu recommendation using TKPI/URT data and MILP optimization",
    version="1.0.0",
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

    day_number: int | None = None
    allowed_food_names: List[str] = Field(default_factory=list)
    excluded_food_names: List[str] = Field(default_factory=list)

    # New: weekly history
    used_food_counts: Dict[str, int] = Field(default_factory=dict)

    # New: weekly max repeat rules
    weekly_rules: Dict[str, int] = Field(
        default_factory=lambda: {
            "LH": 3,
            "LN": 4,
            "S": 3,
            "B": 3,
            "MP": 7,
            "SS": 7,
            "M": 7,
        }
    )


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
        columns=[c for c in old_result_cols if c in df.columns], errors="ignore"
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


def filter_foods_by_weekly_candidates(
    df: pd.DataFrame,
    allowed_food_names: List[str],
    excluded_food_names: List[str],
) -> pd.DataFrame:
    filtered = df.copy()

    if allowed_food_names:
        allowed_set = set(allowed_food_names)
        filtered = filtered[filtered["food_name"].isin(allowed_set)]

    if excluded_food_names:
        excluded_set = set(excluded_food_names)
        filtered = filtered[~filtered["food_name"].isin(excluded_set)]

    return filtered


def get_nearest_energy_level(energy_kcal: int) -> int:
    return min(PORTION_PLAN.keys(), key=lambda x: abs(x - energy_kcal))


def adjust_portion_plan_for_disease(
    base_plan: Dict[str, float], diet_type: str
) -> Dict[str, float]:
    plan = base_plan.copy()

    if "DM" in diet_type:
        plan["G"] = 0
        if "M" in plan:
            plan["M"] = min(plan["M"], 2)

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
        # Always constrain macros for all users
        "carb_min_g": request.carb_g * 0.90,
        "carb_max_g": request.carb_g * 1.10,
        "protein_min_g": request.protein_g * 0.90,
        "protein_max_g": request.protein_g * 1.20,
        "fat_min_g": request.fat_g * 0.80,
        "fat_max_g": request.fat_g * 1.20,
        "sodium_max_mg": None,
        "fiber_min_g": None,
    }

    # Hypertension: sodium must be limited
    if "HT" in request.diet_type:
        constraints["sodium_max_mg"] = request.sodium_mg_max

    # Diabetes: fiber target is important
    if "DM" in request.diet_type:
        constraints["fiber_min_g"] = request.fiber_g_min

    # Obesity: make fat upper bound stricter
    if "OBESITY" in request.diet_type:
        constraints["fat_max_g"] = request.fat_g * 1.20

    return constraints


def build_debug_info(
    milp_df: pd.DataFrame,
    adjusted_plan: Dict[str, float],
    constraints: Dict[str, Any],
    solver_status: str,
) -> Dict[str, Any]:

    category_counts = milp_df.groupby("category_code").size().to_dict()

    category_stats = []

    for category, target_portion in adjusted_plan.items():
        cat_df = milp_df[milp_df["category_code"] == category].copy()

        if cat_df.empty:
            category_stats.append(
                {
                    "category_code": category,
                    "target_portion": target_portion,
                    "available_foods": 0,
                    "min_energy_per_portion": None,
                    "max_energy_per_portion": None,
                    "min_fat_per_portion": None,
                    "max_fat_per_portion": None,
                    "min_carb_per_portion": None,
                    "max_carb_per_portion": None,
                    "min_fiber_per_portion": None,
                    "max_fiber_per_portion": None,
                }
            )
            continue

        category_stats.append(
            {
                "category_code": category,
                "target_portion": target_portion,
                "available_foods": int(len(cat_df)),
                "min_energy_per_portion": float(
                    cat_df["energy_kcal_per_portion"].min()
                ),
                "max_energy_per_portion": float(
                    cat_df["energy_kcal_per_portion"].max()
                ),
                "min_fat_per_portion": float(cat_df["fat_g_per_portion"].min()),
                "max_fat_per_portion": float(cat_df["fat_g_per_portion"].max()),
                "min_carb_per_portion": float(cat_df["carb_g_per_portion"].min()),
                "max_carb_per_portion": float(cat_df["carb_g_per_portion"].max()),
                "min_fiber_per_portion": float(cat_df["fiber_g_per_portion"].min()),
                "max_fiber_per_portion": float(cat_df["fiber_g_per_portion"].max()),
            }
        )

    return {
        "message": "MILP failed to find a feasible menu.",
        "solver_status": solver_status,
        "possible_reason": [
            "The selected diet_type constraints may conflict with the portion plan.",
            "Fiber minimum may be too high for the available food candidates.",
            "Oil/fat portion may be too high if obesity adjustment is not applied.",
            "Fruit/LH/oil variety constraints may make the solution too restrictive.",
            "Some required categories may have too few valid candidate foods.",
        ],
        "constraints": constraints,
        "adjusted_portion_plan": adjusted_plan,
        "category_counts": category_counts,
        "category_stats": category_stats,
    }


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
            milp_df[col] = pd.to_numeric(
                milp_df[col], errors="coerce").fillna(0)

    return milp_df


def get_max_portion_for_category(
    category_code: str, adjusted_plan: Dict[str, float]
) -> float:
    category_code = str(category_code)

    # MP is limited to 1 type, so the selected MP must be allowed
    # to cover the daily MP portion target.
    if category_code == "MP":
        return max(2.0, adjusted_plan.get("MP", 2.0))

    # Vegetables are limited to max 2 types, but portion can be larger.
    if category_code == "S":
        return max(2.0, adjusted_plan.get("S", 2.0))

    # Fruit max 2 types. Usually target is 2, but allow target-based max.
    if category_code == "B":
        return max(2.0, adjusted_plan.get("B", 2.0))

    # LH max 2 types. If target LH = 3, one item may need 1.5 or 2 portions.
    if category_code == "LH":
        return max(2.0, adjusted_plan.get("LH", 2.0))

    # LN max 2 types. Your debug shows LN has only 1 candidate,
    # so it must be allowed to reach LN target.
    if category_code == "LN":
        return max(2.0, adjusted_plan.get("LN", 2.0))

    # Oil/fat may follow adjusted plan.
    if category_code == "M":
        return max(2.0, adjusted_plan.get("M", 2.0))

    if category_code == "G":
        return max(2.0, adjusted_plan.get("G", 2.0))

    if category_code == "SS":
        return max(2.0, adjusted_plan.get("SS", 1.0))

    return MAX_PORTION_PER_FOOD


def solve_milp_menu(
    milp_df: pd.DataFrame,
    adjusted_plan: Dict[str, float],
    constraints: Dict[str, Any],
    request: MenuRequest,
) -> pd.DataFrame:

    model = pulp.LpProblem("Menu_Recommendation", pulp.LpMinimize)

    x = {}

    for i in milp_df.index:
        category_code = str(milp_df.loc[i, "category_code"])

        max_portion_for_food = get_max_portion_for_category(
            category_code, adjusted_plan
        )

        x[i] = pulp.LpVariable(
            f"x_{i}",
            lowBound=0,
            upBound=int(max_portion_for_food / PORTION_STEP),
            cat="Integer",
        )

    portion = {i: x[i] * PORTION_STEP for i in milp_df.index}

    def get_max_x_for_index(i: int) -> int:
        category_code = str(milp_df.loc[i, "category_code"])

        max_portion_for_food = get_max_portion_for_category(
            category_code, adjusted_plan
        )

        return int(max_portion_for_food / PORTION_STEP)

    total_energy = pulp.lpSum(
        portion[i] * milp_df.loc[i, "energy_kcal_per_portion"] for i in milp_df.index
    )

    total_protein = pulp.lpSum(
        portion[i] * milp_df.loc[i, "protein_g_per_portion"] for i in milp_df.index
    )

    total_fat = pulp.lpSum(
        portion[i] * milp_df.loc[i, "fat_g_per_portion"] for i in milp_df.index
    )

    total_carb = pulp.lpSum(
        portion[i] * milp_df.loc[i, "carb_g_per_portion"] for i in milp_df.index
    )

    total_sodium = pulp.lpSum(
        portion[i] * milp_df.loc[i, "sodium_mg_per_portion"] for i in milp_df.index
    )

    total_fiber = pulp.lpSum(
        portion[i] * milp_df.loc[i, "fiber_g_per_portion"] for i in milp_df.index
    )
    # --------------------------------------------------------
    # Portion structure
    # Hard: M, G, SS
    # Soft: others
    # --------------------------------------------------------

    hard_categories = []
    portion_dev = {}

    for category, target_portion in adjusted_plan.items():
        category_indices = milp_df[milp_df["category_code"]
                                   == category].index.tolist()

        if len(category_indices) == 0 and target_portion > 0:
            raise HTTPException(
                status_code=400,
                detail=f"No candidate food for required category {category}",
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
                f"soft_portion_{category}",
            )

            portion_dev[category] = dev_pos + dev_neg

    # --------------------------------------------------------
    # Nutrient constraints
    # --------------------------------------------------------

    energy_target = constraints["energy_target"]
    energy_min = energy_target * 0.95
    energy_max = energy_target * 1.05

    model += total_energy >= energy_min, "energy_min"
    model += total_energy <= energy_max, "energy_max"

    if constraints.get("carb_min_g") is not None:
        model += total_carb >= constraints["carb_min_g"], "carb_min"

    if constraints.get("carb_max_g") is not None:
        model += total_carb <= constraints["carb_max_g"], "carb_max"

    if constraints.get("protein_min_g") is not None:
        model += total_protein >= constraints["protein_min_g"], "protein_min"

    if constraints.get("protein_max_g") is not None:
        model += total_protein <= constraints["protein_max_g"], "protein_max"
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

    veg_indices = milp_df[milp_df["category_code"] == "S"].index.tolist()
    veg_used = {i: pulp.LpVariable(
        f"veg_used_{i}", cat="Binary") for i in veg_indices}

    for i in veg_indices:
        model += x[i] <= get_max_x_for_index(i) * veg_used[i]

    if veg_indices:
        model += (
            pulp.lpSum(veg_used[i] for i in veg_indices) <= 2,
            "max_two_vegetable_types_per_day",
        )
    # Max 1 MP type per day
    mp_indices = milp_df[milp_df["category_code"] == "MP"].index.tolist()
    mp_used = {i: pulp.LpVariable(f"mp_used_{i}", cat="Binary")
               for i in mp_indices}

    for i in mp_indices:
        model += x[i] <= get_max_x_for_index(i) * mp_used[i]

    if mp_indices:
        model += (
            pulp.lpSum(mp_used[i] for i in mp_indices) <= 1,
            "max_one_mp_type_per_day",
        )
    # Max 1 fruit type per day
    fruit_indices = milp_df[milp_df["category_code"] == "B"].index.tolist()
    fruit_used = {
        i: pulp.LpVariable(f"fruit_used_{i}", cat="Binary") for i in fruit_indices
    }

    for i in fruit_indices:
        model += x[i] <= get_max_x_for_index(i) * fruit_used[i]

    if fruit_indices:
        model += (
            pulp.lpSum(fruit_used[i] for i in fruit_indices) <= 2,
            "max_two_fruit_type_per_day",
        )

    # Max 2 LH types per day
    lh_indices = milp_df[milp_df["category_code"] == "LH"].index.tolist()
    lh_used = {i: pulp.LpVariable(f"lh_used_{i}", cat="Binary")
               for i in lh_indices}

    for i in lh_indices:
        model += x[i] <= get_max_x_for_index(i) * lh_used[i]

    if lh_indices:
        model += (
            pulp.lpSum(lh_used[i] for i in lh_indices) <= 2,
            "max_two_lh_types_per_day",
        )

    # Max 1 oil type per day
    oil_indices = milp_df[milp_df["category_code"] == "M"].index.tolist()
    oil_used = {i: pulp.LpVariable(
        f"oil_used_{i}", cat="Binary") for i in oil_indices}

    for i in oil_indices:
        model += x[i] <= get_max_x_for_index(i) * oil_used[i]

    if oil_indices:
        model += (
            pulp.lpSum(oil_used[i] for i in oil_indices) <= 1,
            "max_one_oil_type_per_day",
        )
    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    energy_dev_pos = pulp.LpVariable("energy_dev_pos", lowBound=0)
    energy_dev_neg = pulp.LpVariable("energy_dev_neg", lowBound=0)

    model += (
        total_energy - energy_target == energy_dev_pos - energy_dev_neg,
        "energy_deviation",
    )
    protein_dev_pos = pulp.LpVariable("protein_dev_pos", lowBound=0)
    protein_dev_neg = pulp.LpVariable("protein_dev_neg", lowBound=0)

    fat_dev_pos = pulp.LpVariable("fat_dev_pos", lowBound=0)
    fat_dev_neg = pulp.LpVariable("fat_dev_neg", lowBound=0)

    carb_dev_pos = pulp.LpVariable("carb_dev_pos", lowBound=0)
    carb_dev_neg = pulp.LpVariable("carb_dev_neg", lowBound=0)

    model += (
        total_protein - request.protein_g == protein_dev_pos - protein_dev_neg,
        "protein_deviation",
    )

    model += (
        total_fat - request.fat_g == fat_dev_pos - fat_dev_neg,
        "fat_deviation",
    )

    model += (
        total_carb - request.carb_g == carb_dev_pos - carb_dev_neg,
        "carb_deviation",
    )

    portion_penalty = pulp.lpSum(20 * portion_dev[cat] for cat in portion_dev)

    model += (
        10 * (energy_dev_pos + energy_dev_neg)
        + 5 * (protein_dev_pos + protein_dev_neg)
        + 3 * (carb_dev_pos + carb_dev_neg)
        + 5 * (fat_dev_pos + fat_dev_neg)
        + 0.01 * total_sodium
        + portion_penalty
    ), "objective"

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        debug_info = build_debug_info(
            milp_df=milp_df,
            adjusted_plan=adjusted_plan,
            constraints=constraints,
            solver_status=status,
        )

        raise HTTPException(status_code=422, detail=debug_info)

    selected_items = []

    for i in milp_df.index:
        val = pulp.value(x[i])
        if val is not None and val > 0:
            row = milp_df.loc[i].copy()
            row["selected_portions"] = val * PORTION_STEP
            selected_items.append(row)

    if not selected_items:
        raise HTTPException(
            status_code=422, detail="MILP optimal but no food selected."
        )

    milp_menu = pd.DataFrame(selected_items)

    for col in NUTRIENT_COLS:
        if col in milp_menu.columns:
            total_col = col.replace("_per_portion", "_selected_total")
            milp_menu[total_col] = milp_menu[col] * \
                milp_menu["selected_portions"]

    milp_menu["selected_gram"] = (
        milp_menu["gram_per_portion"] * milp_menu["selected_portions"]
    )

    return milp_menu


# ============================================================
# MEAL ALLOCATION
# ============================================================


def split_preserve_total(
    total_portion: float, ratios: Dict[str, float], step: float
) -> Dict[str, float]:
    meal_names = list(ratios.keys())

    raw = {meal: total_portion * ratio for meal, ratio in ratios.items()}

    rounded = {meal: np.floor(value / step) *
               step for meal, value in raw.items()}

    remaining = round(total_portion - sum(rounded.values()), 10)

    remainders = sorted(
        meal_names, key=lambda meal: raw[meal] - rounded[meal], reverse=True
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

    if lh_df.empty:
        return meal_rows

    lh_df = lh_df.copy().reset_index(drop=True)

    def make_meal_row(row, meal_name, portion_to_assign):
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

        return meal_row

    # Flatten LH into 0.5-step units first
    lh_units = []

    for _, row in lh_df.iterrows():
        remaining = float(row["selected_portions"])

        while remaining >= 0.5 - 1e-9:
            lh_units.append(row.copy())
            remaining = round(remaining - 0.5, 10)

    # Total 0.5 units
    total_half_units = len(lh_units)

    # Target meals in priority order
    target_meals = ["breakfast", "lunch", "dinner"]

    # Case: exactly 3 portions = 6 half-units
    # Allocate 2 half-units = 1 portion to each meal.
    if total_half_units >= 6:
        unit_index = 0

        for meal_name in target_meals:
            row = lh_units[unit_index]
            meal_rows.append(make_meal_row(row, meal_name, 1.0))
            unit_index += 2

        # Any remaining units are added to lunch, not dinner as 0.5 alone
        remaining_units = total_half_units - 6

        if remaining_units > 0:
            extra_portion = remaining_units * 0.5

            # Add extra to lunch row
            lunch_row = meal_rows[1]
            row = lh_units[unit_index]

            lunch_row["meal_portion"] += extra_portion
            lunch_row["meal_gram"] += extra_portion * row["gram_per_portion"]
            lunch_row["meal_urt_qty"] += extra_portion * row["urt_qty"]
            lunch_row["meal_urt"] = f"{lunch_row['meal_urt_qty']} {row['urt_unit']}"

            for col in NUTRIENT_COLS:
                if col in row.index:
                    meal_col = col.replace("_per_portion", "")
                    lunch_row[meal_col] += extra_portion * row[col]

        return meal_rows

    # Case: total LH = 2.5 portions
    if total_half_units == 5:
        row1 = lh_units[0]
        row2 = lh_units[2]

        meal_rows.append(make_meal_row(row1, "breakfast", 1.0))
        meal_rows.append(make_meal_row(row2, "lunch", 1.5))

        return meal_rows

    # Case: total LH = 2 portions
    if total_half_units == 4:
        row1 = lh_units[0]
        row2 = lh_units[2]

        meal_rows.append(make_meal_row(row1, "breakfast", 1.0))
        meal_rows.append(make_meal_row(row2, "lunch", 1.0))

        return meal_rows

    # Case: total LH = 1.5 portions
    if total_half_units == 3:
        row1 = lh_units[0]
        meal_rows.append(make_meal_row(row1, "breakfast", 1.5))

        return meal_rows

    # Case: total LH = 1 portion
    if total_half_units == 2:
        row1 = lh_units[0]
        meal_rows.append(make_meal_row(row1, "breakfast", 1.0))

        return meal_rows

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
        "SS": {
            "morning_snack": 1.0,
        },
        "G": {
            "afternoon_snack": 1.0,
        },
    }

    step_size = {
        "MP": 1.0,
        "LH": 1.0,
        "LN": 1.0,
        "S": 1.0,
        "B": 1.0,
        "M": 0.5,
        "SS": 1.0,
        "G": 1.0,
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


def merge_same_food_in_same_meal(meal_df: pd.DataFrame) -> pd.DataFrame:
    if meal_df.empty:
        return meal_df

    group_cols = [
        "meal_time",
        "food_name",
        "category_code",
        "urt_unit",
        "gram_per_portion",
        "urt_qty",
    ]

    sum_cols = [
        "meal_portion",
        "meal_gram",
        "meal_urt_qty",
        "energy_kcal",
        "protein_g",
        "fat_g",
        "carb_g",
        "sodium_mg",
        "fiber_g",
    ]

    group_cols = [c for c in group_cols if c in meal_df.columns]
    sum_cols = [c for c in sum_cols if c in meal_df.columns]

    merged_df = meal_df.groupby(group_cols, as_index=False)[sum_cols].sum()

    if "meal_urt_qty" in merged_df.columns and "urt_unit" in merged_df.columns:
        merged_df["meal_urt"] = (
            merged_df["meal_urt_qty"].round(2).astype(str)
            + " "
            + merged_df["urt_unit"].astype(str)
        )

    return merged_df


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
            items.append(
                {
                    "food_name": row["food_name"],
                    "category_code": row["category_code"],
                    "portion": float(row["meal_portion"]),
                    "urt": row["meal_urt"],
                    "gram": round(float(row["meal_gram"]), 2),
                    "energy_kcal": round(float(row["energy_kcal"]), 2),
                    "protein_g": round(float(row["protein_g"]), 2),
                    "fat_g": round(float(row["fat_g"]), 2),
                    "carb_g": round(float(row["carb_g"]), 2),
                    "sodium_mg": round(float(row["sodium_mg"]), 2),
                    "fiber_g": round(float(row["fiber_g"]), 2),
                }
            )

        meal_summary = meal_items_df[SUMMARY_COLS].sum().round(2).to_dict()

        meals.append(
            {
                "meal_time": meal_time,
                "summary": meal_summary,
                "items": items,
            }
        )

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
    return {"message": "MILP Menu Recommendation API is running."}


@app.post("/recommend-menu")
def recommend_menu(request: MenuRequest):
    candidate_for_patient = filter_foods_for_disease(
        FOOD_CANDIDATES,
        request.diet_type,
    )

    candidate_for_patient = filter_foods_by_weekly_candidates(
        candidate_for_patient,
        allowed_food_names=request.allowed_food_names,
        excluded_food_names=request.excluded_food_names,
    )

    milp_df = prepare_milp_dataframe(candidate_for_patient)

    if milp_df.empty:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No valid food candidates after applying allowed/excluded food filters.",
                "allowed_food_names_count": len(request.allowed_food_names),
                "excluded_food_names_count": len(request.excluded_food_names),
            },
        )
    nearest_energy = get_nearest_energy_level(request.energy_kcal)
    base_plan = PORTION_PLAN[nearest_energy]

    adjusted_plan = adjust_portion_plan_for_disease(
        base_plan,
        request.diet_type,
    )

    for category in list(adjusted_plan.keys()):
        if category not in milp_df["category_code"].unique():
            adjusted_plan[category] = 0

    constraints = get_disease_constraints(request)

    milp_menu = solve_milp_menu(
        milp_df=milp_df,
        adjusted_plan=adjusted_plan,
        constraints=constraints,
        request=request,
    )

    meal_df = allocate_meals(milp_menu)
    meal_df = merge_same_food_in_same_meal(meal_df)

    return format_response(
        request=request,
        milp_menu=milp_menu,
        meal_df=meal_df,
        adjusted_plan=adjusted_plan,
    )
