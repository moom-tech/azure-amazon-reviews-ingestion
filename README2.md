# Amazon Electronics ETL Pipeline

## Overview
This project implements a **Medallion Architecture** data pipeline using **Azure Databricks** and **Spark**. It processes raw Amazon product reviews and metadata, moving them through Bronze, Silver, and Gold layers.

## Architecture
- **Bronze (Raw):** Ingested JSON data stored in ADLS Gen2.
- **Silver (Processed):** Cleaned data joined with product metadata, stored as **Parquet** for performance.
- **Gold (Curated):** Final feature-engineered dataset ready for analysis and visualization.

## Technologies Used
- **Databricks Jobs:** Orchestrates the multi-step pipeline.
- **PySpark:** Used for distributed data transformations.
- **Parquet:** Chosen for its columnar storage efficiency and fast read/write speeds.
- **Matplotlib/Seaborn:** Used for data insights and brand performance visualization.

## How to Run
1. Configure the ADLS Gen2 mount points in the first notebook.
2. Run the **Databricks Job** `lab3_data_preprocessing_job` to execute notebooks 01 through 03 in sequence.
