# azure-amazon-reviews-ingestion
Data ingestion and transformation pipeline for Amazon Electronics reviews using Azure Data Factory and Blob Storage.
Data Ingestion and Transformation: Amazon Electronics Dataset
Introduction
This lab documents the end-to-end ingestion and transformation of Amazon Electronics review data using Microsoft Azure. The goal was to build a scalable data lake architecture that moves data from a raw, unorganized state into a structured, analytics-ready format using Azure Data Factory.

Objectives
Establish a Data Lake using Azure Blob Storage Gen2 with hierarchical namespaces.

Implement both UI-based and CLI-based ingestion methods to simulate real-world workflows.

Fix semi-structured data formatting issues using Python.

Build an Azure Data Factory (ADF) pipeline to convert JSON data into Parquet and partition it by time.

Project Architecture
The pipeline follows a standard "Medallion" data architecture pattern:

Raw Zone: Stores the original, immutable JSON datasets.

Processed Zone: Stores the optimized Parquet files, cleaned and partitioned for performance.

Implementation Details
1. Data Lake Configuration
I created a Storage Account named amazondatalake60304437 and configured three primary containers: raw, processed, and curated. To ensure efficient file management and security, I enabled Hierarchical Namespaces and used Shared Access Signature (SAS) tokens for terminal-based uploads.

2. Pre-processing & Ingestion
The metadata file (meta_Electronics.json) was originally formatted as Python dictionaries (using single quotes), which is incompatible with many standard JSON parsers.

The Fix: I utilized an Azure ML Compute Instance (VM) to run a Python script using the ast library. This script safely evaluated the Python strings and re-wrote them as valid, line-delimited JSON.

The Upload: While the metadata was uploaded via the Portal UI, the larger reviews dataset (~500MB compressed) was streamed directly to the VM and moved into the raw container using azcopy.

3. Data Factory Pipeline
I built a Mapping Data Flow to handle the ETL (Extract, Transform, Load) logic:

Source: Raw JSON reviews from the storage account.

Transformation: To optimize downstream analytics, I used a Derived Column transformation to extract the year from the unixReviewTime field.

Expression: year(toTimestamp(toLong(unixReviewTime) * 1000))

Sink: The data was written to the processed container in Parquet format.

Partitioning: I configured the sink to "Partition by Key" using the review_year column. This resulted in a folder structure organized by years (1999–2014), which significantly reduces data scanning costs for year-over-year analysis.

4. Automation
To demonstrate operationalizing the pipeline, I implemented a Schedule Trigger. This allows the ingestion process to run automatically on a daily recurrence, ensuring the data lake stays updated without manual intervention.

Conclusion
This lab successfully demonstrates how to bridge the gap between raw web-scraped data and structured cloud storage. By using a combination of Python for data cleaning and Azure Data Factory for orchestration, I created a robust pipeline that prepares data for high-performance analytics.
