try:
    import os
    import logging
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Dict, Any
    from pathlib import Path
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as e:
    missing_module = str(e).split("'")[1]
    raise ImportError(f"Missing required module: {missing_module}. Please install it using 'pip install {missing_module}'.")

# API metadata
app = FastAPI(
    title="Supermarket Sales API",
    description="API for analyzing supermarket sales data and association rules.",
    version="1.0.0"
)

# Load dataset using pathlib
data_file = Path("supermarket_sales.xlsx")  # Ensure the file is in the same directory as main.py
if not data_file.exists():
    raise FileNotFoundError(f"Dataset file '{data_file}' not found. Please place 'supermarket_sales.xlsx' in the 'new_super' directory.")
df = pd.read_excel(data_file)

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
    # ...additional fields if needed...

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
    """
    Retrieve all sales data from the dataset.
    """
    return df.to_dict(orient="records")

# 2. Get unique products
@app.get("/products", response_model=List[str])
def get_unique_products() -> List[str]:
    """
    Retrieve a list of unique product names.
    """
    return df["Product"].unique().tolist()

# 3. Filter sales by product name
@app.get("/sales/product", response_model=List[SalesData])
def get_sales_by_product(product: str = Query(..., description="Enter product name")) -> List[Dict[str, Any]]:
    """
    Retrieve sales data filtered by a specific product name.
    """
    if product not in df["Product"].unique():
        raise HTTPException(status_code=404, detail=f"Product '{product}' not found.")
    filtered_sales = df[df["Product"] == product]
    return filtered_sales.to_dict(orient="records")

# 4. Get top-selling products
@app.get("/sales/top-products", response_model=List[TopProducts])
def get_top_products(n: int = Query(5, ge=1, le=50, description="Number of top products to retrieve (1-50)")) -> List[Dict[str, Any]]:
    """
    Retrieve the top N selling products based on quantity.
    """
    top_products = df.groupby("Product")["Quantity"].sum().nlargest(n)
    return [{"Product": product, "Quantity": quantity} for product, quantity in top_products.items()]

# 5. Get association rules
@app.get("/rules", response_model=List[AssociationRule])
def get_association_rules() -> List[Dict[str, Any]]:
    """
    Retrieve association rules generated from the Apriori algorithm.
    """
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred."},
    )

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting the application...")
    uvicorn.run(app, host="127.0.0.1", port=8000)