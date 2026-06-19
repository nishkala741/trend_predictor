This version seamlessly maps your interactive Python application code (app.py) to the 1M+ record Myntra Products Dataset by Ronak Bokaria, explicitly tracking fields like price, mrp, rating, ratingTotal, and the extracted seller/brand categories to power your price tiers and popularity metrics.

Your operating system will open a local web server pipeline rendering the interface at http://localhost:8501.

🧠 Data Processing & Architecture Blueprint
The underlying analytic engine reads core properties directly sourced from the Kaggle Myntra Products Dataset: https://www.kaggle.com/datasets/ronakbokaria/myntra-products-dataset

Price_Tier Engineering Strategy: Standardizing multi-bracket item prices into 5 quantile bins (Budget, Value, Mid-Range, Premium, and Luxury) by executing standard Pandas distribution splits on raw price attributes.

Popularity_Bin Mathematical Thresholding: Raw transaction/interaction volume counts (ratingTotal) are scaled to assign consumer engagement bins: Niche (<100), Growing (100-999), Popular (1K-5K), Trending (5K-10K), and Viral (10K+).

Discount and Profit Margin Extraction: Leverages discrepancies between raw retail markers (mrp minus price) to map structural margins against user sentiment scores across distinct product categories.

📜 Complete Technical Stack
UI Hosting & Interactive Views: Streamlit Open-Source Framework

Deep Learning Context Engines: Hugging Face Inference Pipelines (DistilBERT Architecture), PyTorch Runtime Core

Linguistic Analysis Fallbacks: TextBlob Sentiment Processors, Native Regular Expression Tokenizers

High-Volume Matrix Engineering: Pandas DataFrames, NumPy Arrays

Vector Visualization Engines: Plotly Express (Interactive Hex/Bar Elements), Plotly Graph Objects, Seaborn, Matplotlib Core
