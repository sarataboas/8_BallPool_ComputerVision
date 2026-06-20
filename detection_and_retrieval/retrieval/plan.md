# Task 3.2

## **Image Retrieval**

Query image -> Feature extraction -> Similarity computation -> Top-K most similar images (ordered - similarity score)

**Goal**: Retrieve images where the balls occupy similar positions relative to the table, independently of the viewpoint
Similarity -> balls located in similar positions relative to the table, independently of the view


#### **Methods:** 

1. Pixel Comparison (baseline notebook):
- Mean Squared Error 
- Structural Similarity

Pixel comparison tends to fail because different cameras on the same table will produce huge pixel differences.


2. Embeddings
- Idea: Image -> Neural Network -> Vector (embedding) and compare the embeddings of the query against the embeddings of the database



## **Retrieval Pipeline**

1. Offline Phase: Compute the embeddings for all the training data images and store them

2. Query Phase: Query image -> embedding -> compare to all stored embeddings -> sort distances -> return top-k


## **Structure**
```bash
retrieval/
│
├── utils/
│   ├── data.py          # load partition.csv, load images, split train/test
│   ├── metrics.py       # MSE, SSIM, cosine similarity
│   ├── features.py      # ResNet embedding extraction
│   └── visualization.py # plot query + top-k results
│
├── experiment_1_raw_baseline.ipynb
├── experiment_2_topview.ipynb
├── experiment_3_contrastive.ipynb
└── experiment_4_combined.ipynb
```


## **Experiments**

#### **Experiments Overview**
| Experiment  | Brief Description | Status  | Results File | Observations | 
|------------|-------------------|--------|---------------------------|--------|
| E1 | Raw Image Retrieval Baseline | [-] Progress  | - | - |



### **Experiment 1: Raw Image Retrieval Baseline**

- **Goal:** Establish a raw-image retrieval baseline and analyze whether each method retrieves images with similar table states or is biased toward similar viewpoints.
**Input:**
- query images: test
- retrieval database: train
- split from partition.csv

**Methods:**
Use the original images directly: 
- Mean-Squared Error (MSE)
- Structural Similarity (SSIM)
- Pretrained ResNet embeddings + cosine similarity


**Output:**
For each query image:
- Query image
- Top-5 retrieved images with MSE
- Top-5 retrieved images with SSIM
- Top-5 retrieved images with ResNet

**Expected:** 
- MSE and SSIM will probably retrieve images with similar camera pose / similar colors instead of same or similar table state
- ResNet should be better, but still may suffer from viewpoint bias.