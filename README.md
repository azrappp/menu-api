# Menu API

A FastAPI-based menu recommendation API using **MILP optimization** to generate daily meal recommendations based on user energy needs and nutrition constraints.

This project is designed as a prototype for dietary recommendation using Indonesian food data, TKPI nutrient values, URT household measurements, and expert-based portion planning.

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Pandas
- NumPy
- PuLP

---

## Project Structure

```text
menu-api/
├── main.py
├── ready_food_data_frame.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Setup

1. Clone Repository
   git clone git@github.com:azrappp/menu-api.git
   cd menu-api
2. Create Virtual Environment
   Windows
   python -m venv venv
   venv\Scripts\activate
   Mac / Linux
   python3 -m venv venv
   source venv/bin/activate
3. Install Dependencies
   pip install -r requirements.txt

## API Endpoint

POST /recommend-menu

Generate daily menu recommendation.

Request Body
{
"energy_kcal": 1600,
"diet_type": "DM_HT_OBESITY",
"carb_g": 240,
"protein_g": 60,
"fat_g": 44,
"sodium_mg_max": 2000,
"fiber_g_min": 25
}
Response Example
{
"status": "success",
"message": "Menu recommendation generated successfully.",
"input": {
"diet_type": "DM_HT_OBESITY",
"energy_kcal": 1600,
"carb_g": 240,
"protein_g": 60,
"fat_g": 44,
"sodium_mg_max": 2000,
"fiber_g_min": 25
},
"adjusted_portion_plan": {
"MP": 4,
"LH": 3,
"LN": 3,
"S": 3,
"B": 2,
"SS": 0,
"M": 2,
"G": 0
},
"daily_total": {
"energy_kcal": 1580.0,
"protein_g": 70.0,
"fat_g": 48.0,
"carb_g": 225.0,
"sodium_mg": 220.0,
"fiber_g": 25.0
},
"meals": [
{
"meal_time": "breakfast",
"summary": {
"energy_kcal": 367.0,
"protein_g": 10.69,
"fat_g": 2.08,
"carb_g": 74.63,
"sodium_mg": 11.5,
"fiber_g": 0.5
},
"items": [
{
"food_name": "Nasi",
"category_code": "MP",
"portion": 1.0,
"urt": "0.75 Gelas",
"gram": 100.0,
"energy_kcal": 180.0,
"protein_g": 3.0,
"fat_g": 0.3,
"carb_g": 39.8,
"fiber_g": 0.2
}
]
}
]
}
