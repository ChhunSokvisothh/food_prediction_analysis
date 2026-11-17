import joblib

commodity_to_code = joblib.load("commodity_encoder.pkl")
province_to_code = joblib.load("province_encoder.pkl")
reference_stats = joblib.load("reference_stats.pkl")

print("COMMODITIES:")
for c in sorted(commodity_to_code.keys()):
    print(f"  {c}")

print("\nPROVINCES:")
for p in sorted(province_to_code.keys()):
    print(f"  {p}")

print("\nREFERENCE STATS INDEX:")
print(reference_stats.index.tolist()[:10])
