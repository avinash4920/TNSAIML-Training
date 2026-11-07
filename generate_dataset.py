"""Generate a clearly labelled demo housing dataset for local project testing."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    rng = np.random.default_rng(42)
    n = 1500
    locations = np.array(["Hyderabad", "Bengaluru", "Chennai", "Pune", "Mumbai", "Delhi", "Other"])
    location = rng.choice(locations, n, p=[.25, .18, .14, .12, .12, .09, .10])
    area = rng.integers(450, 4001, n)
    bedrooms = np.clip(np.rint(area / rng.uniform(450, 700, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 6)
    floors = rng.integers(1, 6, n)
    age = rng.integers(0, 36, n)
    parking = rng.integers(0, 3, n)
    location_rate = {"Hyderabad": 0.010, "Bengaluru": 0.012, "Chennai": 0.011, "Pune": 0.0105, "Mumbai": 0.020, "Delhi": 0.015, "Other": 0.007}
    base = np.array([location_rate[x] for x in location])
    price = (area * base + bedrooms * 2.2 + bathrooms * 1.2 + floors * 0.8 + parking * 1.5 - age * 0.12 + rng.normal(0, 4, n))
    price = np.clip(price, 12, None)
    df = pd.DataFrame({"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                       "floors": floors, "age_years": age, "parking": parking,
                       "location": location, "price_lakhs": np.round(price, 2)})
    out = ROOT / "data" / "house_prices.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Created {out} with {len(df)} demo records.")
    print("NOTE: This is synthetic demonstration data, not actual market transaction data.")

if __name__ == "__main__":
    main()
