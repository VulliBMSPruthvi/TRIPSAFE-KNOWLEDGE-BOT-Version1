import os
import pandas as pd

pricing_path = r"C:\Users\Pruthvi Vulli\pricing_csvs"  # Update to your folder

def generate_quote(query):
    import re
    try:
        region = "Asia" if "asia" in query.lower() else (
            "Including USA & Canada" if "usa" in query.lower() or "canada" in query.lower() else "Excluding USA & Canada"
        )
        days = int(re.search(r"(\d+)\s*day", query).group(1))
        age = int(re.search(r"(\d+)\s*(yr|year|age)", query).group(1))
        plan = "100k" if "100" in query else "50k"

        filename = f"pricing_{plan}_{region}.csv"
        file_path = os.path.join(pricing_path, filename)

        if not os.path.exists(file_path):
            return f"❌ Could not find pricing file for {plan} plan in region '{region}'"

        df = pd.read_csv(file_path, index_col=0)
        nearest_day = min(df.index, key=lambda x: abs(x - days))
        nearest_age = min(df.columns.astype(int), key=lambda x: abs(x - age))

        price = df.loc[nearest_day, str(nearest_age)]
        return f"💰 Quotation:\n- Region: {region}\n- Days: {nearest_day}\n- Age: {nearest_age}\n- Plan: {plan}\n- **Price: ₹{price}**"

    except Exception as e:
        return f"⚠️ Failed to generate quotation: {str(e)}"
