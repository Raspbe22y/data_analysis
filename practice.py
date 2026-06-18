import pandas as pd

# df_list = [pd.read_csv("breathing_df_cleaned_hour_2.csv"),
#            pd.read_csv("exposure_df_cleaned_hour_2.csv")]
# df_merge = pd.merge(df_list[0], df_list[1], how="outer",
#                     on=["patient_id", "season", "timestamp"])
# df_merge.to_csv("merged_br_exp.csv", index=False)

data = {
    "Product": ["Laptop", "Laptop", "Mobile"],
    "Region": ["North", "North", "South"],
    "Sales": [1000, 1500, 2000],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Group by 'Product' and 'Region' and calculate the sum of 'Sales'
# result = df.groupby(["Product", "Region"])
result = df.groupby(["Product", "Region"])["Sales"]


print(result)
