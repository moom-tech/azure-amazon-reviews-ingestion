# Amazon Reviews Feature Engineering Pipeline

A comprehensive Azure ML feature engineering pipeline for Amazon product reviews. This project extracts multiple types of features from review text and metadata to create a rich dataset for downstream machine learning tasks.

## 📋 Project Overview

This repository contains a complete feature engineering workflow that transforms raw Amazon product reviews into engineered features using Azure ML components and pipelines. The pipeline processes review text to extract:

- **Text Embeddings**: SBERT (Sentence-BERT) pre-trained embeddings
- **Sentiment Analysis**: VADER sentiment scores (positive, negative, neutral, compound)
- **Text Features**: Review length, TF-IDF vectors, normalized text
- **Metadata Features**: Extracted from product and review metadata

### Architecture

```
Raw Data
   ↓
Split (Train/Val/Test)
   ↓
Normalize Text (all splits)
   ↓
Feature Extraction (parallel):
   ├─ Review Length
   ├─ Sentiment Analysis (VADER)
   ├─ SBERT Embeddings
   ├─ TF-IDF Features
   └─ Metadata Extraction
   ↓
Merge Features
   ↓
Feature Store (Amazon Review Text Features)
```

## 🗂️ Project Structure

```
Users/60304437/
├── components/              # Azure ML components (reusable pipeline steps)
│   ├── embeddings/         # SBERT embeddings generator
│   ├── merge/              # Feature merging component
│   ├── metadata/           # Metadata feature extraction
│   ├── normalize_text/     # Text normalization
│   ├── review_length/      # Review length feature
│   ├── sentiment_analysis/ # VADER sentiment analysis
│   ├── split_dataset/      # Train/Val/Test split
│   └── tfidf/              # TF-IDF feature extraction
│
├── pipelines/              # Azure ML pipeline definitions
│   └── feature_pipeline.yaml  # Main feature engineering pipeline
│
├── feature_store/          # Feature store specifications
│   ├── amazon_review_text_features.yaml  # Feature set definition
│   ├── entity_amazon_review.yaml         # Entity definition
│   └── FeatureSetSpec.yaml              # Feature schema
│
├── datastores/             # Datastore configurations
│   └── curated_adls.yaml   # Azure Data Lake Store connection
│
└── data/                   # Sample data and configurations
    └── features_v1_sampled.yaml  # Data asset reference
```

## 🔧 Components

### 1. **split_dataset**
Splits raw review data into train, validation, and test sets to prevent data leakage.

**Input**: Raw review data (parquet)  
**Output**: train, val, test datasets (parquet)  
**Script**: `split.py`

### 2. **normalize_text**
Cleans and normalizes review text by removing special characters, converting to lowercase, and handling missing values.

**Input**: Review dataset with `reviewText` column  
**Output**: Normalized text data (parquet)  
**Script**: `normalize.py`

### 3. **review_length**
Extracts review length as a numerical feature for each review.

**Input**: Review data  
**Output**: Data with `review_length` column (parquet)  
**Script**: `review_length.py`

### 4. **sentiment_analysis**
Analyzes sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

**Features Generated**:
- `sentiment_pos`: Positive sentiment score (0-1)
- `sentiment_neg`: Negative sentiment score (0-1)
- `sentiment_neu`: Neutral sentiment score (0-1)
- `sentiment_compound`: Compound sentiment score (-1 to 1)

**Input**: Review data with `reviewText` column  
**Output**: Data with sentiment columns (parquet)  
**Script**: `sentiment.py`

### 5. **embeddings**
Generates dense vector embeddings using Sentence-BERT (all-MiniLM-L6-v2 model).

**Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Fast, lightweight, high-quality
- Pre-trained on semantic similarity

**Output Features**: `emb_0` to `emb_383` (384 embedding dimensions)

**Input**: Review data with `reviewText` column  
**Output**: Data with embedding columns (parquet)  
**Script**: `embedding.py`

### 6. **tfidf**
Extracts TF-IDF (Term Frequency-Inverse Document Frequency) features from normalized text.

**Input**: Review data with normalized text  
**Output**: Data with TF-IDF sparse vectors (parquet)  
**Script**: `tfidf_features.py`

### 7. **metadata**
Extracts and engineers features from product and review metadata.

**Input**: Review data with metadata columns  
**Output**: Data with engineered metadata features (parquet)  
**Script**: `metadata_features.py`

### 8. **merge**
Combines all extracted features into a single feature-rich dataset.

**Input**: Features from all extraction components  
**Output**: Merged feature dataset (parquet)  
**Script**: `merge_features.py`

## 🚀 Feature Pipeline

The main feature engineering workflow is defined in `feature_pipeline.yaml`.

**Pipeline Steps**:
1. **Split**: Divide data into train/val/test (80/10/10)
2. **Normalize**: Clean text across all three splits
3. **Extract Features** (parallel execution):
   - Generate review length features
   - Compute VADER sentiment scores
   - Create SBERT embeddings
   - Generate TF-IDF vectors
   - Extract metadata features
4. **Merge**: Combine all features into unified dataset

**Compute**: Runs on Azure ML CPU cluster (`cpu-cluster`)

## 📊 Feature Store

The feature store (`feature_store/`) provides structured access to engineered features:

- **Feature Set**: `amazon_review_text_features`
- **Entity**: `amazon_product_entity` (identifies reviews by product/review ID)
- **Schema**: Defined in `FeatureSetSpec.yaml` with feature types and statistics

This enables:
- Versioned feature access
- Feature lineage tracking
- On-demand feature retrieval for model training

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- Azure ML SDK v2
- Required packages (see `conda.yaml` files in each component)

### Key Dependencies

- `sentence-transformers`: SBERT embeddings
- `vaderSentiment`: Sentiment analysis
- `scikit-learn`: TF-IDF vectorization
- `pandas`: Data manipulation
- `pyarrow`: Parquet I/O

### Installation

```bash
# Clone the repository
git clone https://github.com/moom-tech/azure-amazon-reviews-ingestion.git
cd azure-amazon-reviews-ingestion

# Switch to feature engineering branch
git checkout lab4_feature_engineering

# Install Azure ML SDK
pip install azure-ai-ml

# Install component dependencies
pip install sentence-transformers vaderSentiment scikit-learn pandas pyarrow
```

## 💻 Usage

### Run the Feature Pipeline

```bash
# Authenticate with Azure ML
az login
az account set --subscription <subscription-id>

# Run the pipeline
az ml job create --file Users/60304437/pipelines/feature_pipeline.yaml \
  --workspace-name <workspace-name> \
  --resource-group <resource-group>
```

### Access Features from Feature Store

```python
from azure.ai.ml.entities import FeatureSet
from azure.ai.ml import MLClient

# Initialize ML client
ml_client = MLClient(...)

# Get feature set
feature_set = ml_client.feature_sets.get(
    name="amazon_review_text_features",
    version="1"
)

# Retrieve features for specific entities
features = ml_client.feature_sets.get_features_data(
    feature_set=feature_set,
    entity_ids=["review_id_1", "review_id_2", ...]
)
```

## 📈 Data Flow Example

**Input**: Raw Amazon review JSON
```json
{
  "reviewText": "This product is amazing! Highly recommended.",
  "rating": 5,
  "productId": "B00123ABC",
  "timestamp": 1234567890
}
```

**After Feature Engineering**: Rich feature vector
```
reviewText: "this product is amazing highly recommended"  (normalized)
review_length: 43
sentiment_pos: 0.62
sentiment_neg: 0.0
sentiment_neu: 0.38
sentiment_compound: 0.84
emb_0: 0.125  ... emb_383: -0.089  (384 embedding dimensions)
tfidf_amazing: 0.35
tfidf_product: 0.22
...
```

## 🔍 Component Specifications

Each component includes:

- **component.yaml**: Azure ML component metadata (inputs, outputs, compute, environment)
- **conda.yaml**: Python environment dependencies
- **{script}.py**: Implementation logic

All components follow a consistent interface:
- Input: `--data` (path to input parquet)
- Output: `--output` (directory for output parquet)

## 📝 Notes

- **Data Leakage Prevention**: Training data is split before feature extraction to prevent leakage
- **Embeddings**: Uses lightweight SBERT model (384-dim) for fast inference
- **Sentiment Analysis**: VADER is optimized for social media/review text
- **Scalability**: Pipeline components run in parallel where possible
- **Reproducibility**: All models are deterministic (no random seeds in embeddings)

## 🤝 Contributing

To add new features:

1. Create a new component folder under `components/`
2. Implement the feature extraction logic in a Python script
3. Add `component.yaml` and `conda.yaml` with dependencies
4. Update the pipeline YAML to include the new component
5. Test on sample data before running full pipeline

## 📄 License

Specify your license here.

## 📞 Support

For issues or questions:
- Check Azure ML documentation: https://learn.microsoft.com/azure/machine-learning/
- Review component logs in Azure ML Studio
- Ensure all input data columns are present and properly formatted

---

**Last Updated**: March 2026  
**Branch**: `lab4_feature_engineering`
