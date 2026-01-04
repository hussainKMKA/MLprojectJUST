#importing libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
OUT_DIR = r"C:\PATH\TO\output_folder"
os.makedirs(OUT_DIR, exist_ok=True)
#%%
aisles = pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\aisles.csv",dtype={
        "aisle_id": "int32",
        "aisle": "string"})
departments= pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\departments.csv",dtype={
        "department_id": "int32",
        "department": "string"})
order_products__prior= pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\order_products__prior.csv",dtype={
        "order_id": "Int64",
        "product_id": "Int64",
        "add_to_cart_order": "Int16",
        "reordered": "Int8"})
order_products__train=  pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\order_products__train.csv",dtype={
        "order_id": "Int64",
        "product_id": "Int64",
        "add_to_cart_order": "Int16",
        "reordered": "Int8"})
orders= pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\orders.csv",dtype={
        "order_id": "Int64",
        "user_id": "Int64",
        "eval_set": "string",
        "order_number": "Int32",
        "order_dow": "Int16",
        "order_hour_of_day": "Int16",
        "days_since_prior_order": "float32"})
products= pd.read_csv(r"C:\Users\aqsa\OneDrive\Desktop\ML\products.csv",dtype={
        "product_id": "int64",
        "aisle_id": "int32",
        "department_id": "int32",
        "product_name": "string"})
#%%
print("orders:", orders.shape)
print("products:", products.shape)
print("aisles:", aisles.shape)
print("departments:", departments.shape)
print("order_products__prior:", order_products__prior.shape)
print("order_products__train:", order_products__train.shape)

#%%
for df in [orders, products, aisles, departments, order_products__prior, order_products__train]:
    df.columns = df.columns.str.strip()
#%%
product_meta = (
    products
    .merge(aisles, on="aisle_id", how="left", validate="m:1")
    .merge(departments, on="department_id", how="left", validate="m:1")
)

print("product_meta:", product_meta.shape)
#%%
order_products__prior["source"] = "prior"
order_products__train["source"] = "train"

order_products_all = pd.concat([order_products__prior, order_products__train], ignore_index=True)
order_products_all["source"] = order_products_all["source"].astype("category")

print("order_products_all:", order_products_all.shape)
#%%
orders["order_id_key"] = pd.to_numeric(orders["order_id"], errors="coerce")
order_products_all["order_id_key"] = pd.to_numeric(order_products_all["order_id"], errors="coerce")

product_meta["product_id_key"] = pd.to_numeric(product_meta["product_id"], errors="coerce")
order_products_all["product_id_key"] = pd.to_numeric(order_products_all["product_id"], errors="coerce")

orders = orders.dropna(subset=["order_id_key"]).copy()
product_meta = product_meta.dropna(subset=["product_id_key"]).copy()
order_products_all = order_products_all.dropna(subset=["order_id_key", "product_id_key"]).copy()

orders["order_id_key"] = orders["order_id_key"].astype("int64")
product_meta["product_id_key"] = product_meta["product_id_key"].astype("int64")
order_products_all["order_id_key"] = order_products_all["order_id_key"].astype("int64")
order_products_all["product_id_key"] = order_products_all["product_id_key"].astype("int64")
#%%
df = (
    order_products_all
    .merge(orders, on="order_id_key", how="left", validate="m:1")
    .merge(product_meta, on="product_id_key", how="left", validate="m:1")
)

#%%
#lets start doing EDA for the whole dataset and basic vaisualizations

print("shape:", df.shape)

print("unique users:", df["user_id"].nunique())
print("unique orders:", df["order_id_key"].nunique())
print("unique products:", df["product_id_key"].nunique())
df.head(10)
#%%
missing_frac = df.isna().mean().sort_values(ascending=False)
missing_frac = missing_frac[missing_frac > 0]

print("Columns that have missing values:\n", missing_frac)
#%%
counts = df["reordered"].value_counts()
ratios = df["reordered"].value_counts(normalize=True)

print("Counts:\n", counts)
print("\nRatios:\n", ratios)

plt.figure(figsize=(5,4))
counts.plot(kind="bar")
plt.title("Target Distribution: reordered")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(8,4))
df["days_since_prior_order"].fillna(0).plot(kind="hist", bins=50)
plt.title("Distribution: days_since_prior_order (NaN filled with 0 for plot)")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(8,4))
df["add_to_cart_order"].dropna().plot(kind="hist", bins=50)
plt.title("Distribution: add_to_cart_order")
plt.tight_layout()
plt.show()
#%%
cart_size = df.groupby("order_id_key").size()

print(cart_size.describe())

plt.figure(figsize=(8,4))
cart_size.plot(kind="hist", bins=60)
plt.title("Basket Size per Order (number of items)")
plt.tight_layout()
plt.show()

#%%
top_aisles = df["aisle"].value_counts().head(20)

plt.figure(figsize=(12,4))
top_aisles.plot(kind="bar")
plt.title("Top 20 aisles by item count")
plt.tight_layout()
plt.show()
#%%
dept_counts = df["department"].value_counts()

plt.figure(figsize=(10,4))
dept_counts.plot(kind="bar")
plt.title("Department Distribution")
plt.tight_layout()
plt.show()
#%%
hour_counts = df["order_hour_of_day"].value_counts().sort_index()
dow_counts  = df["order_dow"].value_counts().sort_index()

plt.figure(figsize=(12,4))
hour_counts.plot(kind="bar")
plt.title("Orders by hour of day")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
dow_counts.plot(kind="bar")
plt.title("Orders by day of week 0=Sunday")
plt.tight_layout()
plt.show()
#%%
reorder_by_hour = df.groupby("order_hour_of_day")["reordered"].mean()
reorder_by_dow  = df.groupby("order_dow")["reordered"].mean()

plt.figure(figsize=(10,4))
reorder_by_hour.plot(marker="o")
plt.title("Reorder Rate by Hour of Day")
plt.ylabel("Mean reordered")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
reorder_by_dow.plot(marker="o")
plt.title("Reorder Rate by Day of Week")
plt.ylabel("Mean reordered")
plt.tight_layout()
plt.show()
#%%
numeric_cols = [
    "add_to_cart_order",
    "days_since_prior_order",
    "order_hour_of_day",
    "order_dow",
    "order_number",
    "reordered"
]

corr_df = df[numeric_cols].copy()
corr_df["days_since_prior_order"] = corr_df["days_since_prior_order"].fillna(0)

corr_matrix = corr_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap features", fontsize=14)
plt.tight_layout()
plt.show()