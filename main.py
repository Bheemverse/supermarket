import os
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# API metadata
app = FastAPI(
    title="Supermarket Sales API",
    description="API for analyzing supermarket sales data and association rules.",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset using pathlib with Railway compatibility
data_file = Path(os.getenv("DATA_FILE", "supermarket_sales.xlsx"))  # Allow environment variable

if not data_file.exists():
    logging.error(f"Dataset file '{data_file}' not found. Ensure it's uploaded.")
    raise FileNotFoundError(f"Dataset file '{data_file}' not found.")

try:
    df = pd.read_excel(data_file, engine="openpyxl")  # Specify engine for better compatibility
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

# Convert transactions into a basket format
basket = df.groupby(['Invoice ID', 'Product'])['Quantity'].sum().unstack().fillna(0)
basket = (basket > 0).astype(int)

# Apply Apriori Algorithm
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Response models
class SalesData(BaseModel):
    Invoice_ID: str
    Product: str
    Quantity: int

class TopProducts(BaseModel):
    Product: str
    Quantity: int

class AssociationRule(BaseModel):
    antecedents: List[str]
    consequents: List[str]
    support: float
    confidence: float
    lift: float

# 1. Get all sales data
@app.get("/sales", response_model=List[SalesData])
def get_sales_data() -> List[Dict[str, Any]]:
    return df.to_dict(orient="records")

# 2. Get unique products
@app.get("/products", response_model=List[str])
def get_unique_products() -> List[str]:
    return df["Product"].unique().tolist()

# 3. Filter sales by product name
@app.get("/sales/product", response_model=List[SalesData])
def get_sales_by_product(product: str = Query(..., description="Enter product name")) -> List[Dict[str, Any]]:
    if product not in df["Product"].unique():
        raise HTTPException(status_code=404, detail=f"Product '{product}' not found.")
    filtered_sales = df[df["Product"] == product]
    return filtered_sales.to_dict(orient="records")

# 4. Get top-selling products
@app.get("/sales/top-products", response_model=List[TopProducts])
def get_top_products(n: int = Query(5, ge=1, le=50, description="Number of top products to retrieve (1-50)")) -> List[Dict[str, Any]]:
    top_products = df.groupby("Product")["Quantity"].sum().nlargest(n)
    return [{"Product": product, "Quantity": quantity} for product, quantity in top_products.items()]

# 5. Get association rules
@app.get("/rules", response_model=List[AssociationRule])
def get_association_rules() -> List[Dict[str, Any]]:
    rules_dict = rules.to_dict(orient="records")
    formatted_rules = [
        {
            "antecedents": list(rule["antecedents"]),
            "consequents": list(rule["consequents"]),
            "support": rule["support"],
            "confidence": rule["confidence"],
            "lift": rule["lift"]
        }
        for rule in rules_dict
    ]
    return formatted_rules

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )

if __name__ == "__main__":
    import uvicorn

    # Get the correct PORT for Railway
    port = int(os.getenv("PORT", 8000))  # Use Railway's assigned port
    logging.info(f"Starting the application on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)  # Use 0.0.0.0 for external access
