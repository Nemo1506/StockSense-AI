import csv
from datetime import datetime, timedelta
import random

random.seed(42)

products = [
    {
        "id": "PROD_001",
        "name": "Sparkling Mineral Water 1L",
        "category": "Beverages",
        "base_sales": 28,
        "price": 2.50,
        "current_stock_store1": 25,  # Low stock -> High Risk
        "current_stock_store2": 30,
        "cost": 1.20,
    },
    {
        "id": "PROD_002",
        "name": "Greek Whole Milk Yogurt 500g",
        "category": "Dairy",
        "base_sales": 18,
        "price": 4.20,
        "current_stock_store1": 110, # Moderate -> Medium Risk
        "current_stock_store2": 125,
        "cost": 2.40,
    },
    {
        "id": "PROD_003",
        "name": "Hass Avocados 4pk",
        "category": "Produce",
        "base_sales": 14,
        "price": 3.99,
        "current_stock_store1": 320, # High stock -> Low Risk
        "current_stock_store2": 350,
        "cost": 2.10,
    },
    {
        "id": "PROD_004",
        "name": "Artisan Sourdough Loaf",
        "category": "Bakery",
        "base_sales": 10,
        "price": 5.50,
        "current_stock_store1": 15,  # Low stock
        "current_stock_store2": 20,
        "cost": 2.75,
    },
    {
        "id": "PROD_005",
        "name": "Raw Wildflower Honey 500g",
        "category": "Pantry",
        "base_sales": 2,   # Low turnover, intermittent zeroes
        "price": 12.00,
        "current_stock_store1": 45,  # Ample stock -> Low Risk
        "current_stock_store2": 50,
        "cost": 6.50,
    },
]

stores = [
    {"id": "STORE_01", "region": "North"},
    {"id": "STORE_02", "region": "East"}
]

weathers = ["Sunny", "Rainy", "Cloudy", "Overcast"]
start_date = datetime(2026, 8, 1)
days_count = 45

rows = []

for day_idx in range(days_count):
    cur_date = start_date + timedelta(days=day_idx)
    date_str = cur_date.strftime("%Y-%m-%d")
    is_weekend = cur_date.weekday() >= 5
    weather = random.choice(weathers)
    holiday = 1 if day_idx in [15, 30] else 0

    for store in stores:
        for prod in products:
            # Sales with weekend lift and random noise
            mult = 1.35 if is_weekend else 1.0
            if prod["id"] == "PROD_005":
                # Sparse sales with occasional 0
                units_sold = random.choice([0, 1, 2, 3, 0, 1, 4])
            else:
                noise = random.randint(-4, 5)
                units_sold = max(0, int(prod["base_sales"] * mult + noise))

            # Inventory evolves: decreases with sales, occasionally replenished
            # For the latest days, we want inventory to match the test scenario
            if day_idx == days_count - 1:
                inv_level = prod["current_stock_store1"] if store["id"] == "STORE_01" else prod["current_stock_store2"]
            else:
                inv_level = max(5, int(prod["base_sales"] * 7 + random.randint(10, 80) - units_sold))

            units_ordered = 50 if (day_idx % 7 == 0 and inv_level < 50) else 0
            demand_forecast_baseline = max(0, int(units_sold + random.randint(-2, 3)))
            discount = 0.10 if (is_weekend and prod["category"] == "Bakery") else 0.0

            rows.append({
                "Date": date_str,
                "Store_ID": store["id"],
                "Product_ID": prod["id"],
                "Category": prod["category"],
                "Region": store["region"],
                "Inventory_Level": inv_level,
                "Units_Sold": units_sold,
                "Units_Ordered": units_ordered,
                "Demand_Forecast": demand_forecast_baseline,
                "Price": prod["price"],
                "Discount": discount,
                "Weather_Condition": weather,
                "Holiday_Promotion": holiday
            })

with open("data/sample_inventory.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "Date", "Store_ID", "Product_ID", "Category", "Region",
        "Inventory_Level", "Units_Sold", "Units_Ordered", "Demand_Forecast",
        "Price", "Discount", "Weather_Condition", "Holiday_Promotion"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} sample records in data/sample_inventory.csv")
