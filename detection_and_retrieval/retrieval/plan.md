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
│   ├── data.py          # load partition.csv, load images, show images, denormalize images for visualization
├── exp1_raw_baseline.ipynb
├──
```


## **Experiments**

#### **Experiments Overview**
| Experiment  | Brief Description | Status  | Results | Observations | 
|------------|-------------------|--------|---------------------------|--------|
| E1 | Raw Image Retrieval Baseline | [x] Done  | [exp1_raw_baseline](experiments/exp1_raw_baseline.ipynb) | Based on the professor's baseline notebook |
| E2 | Deep Feature Retrieval: Pretrained Embeddings + Cosine Similarity | [-] Progress | - | - |



### **Experiment 1: Raw Image Retrieval Baseline**

- **Goal:** Establish a raw-image retrieval baseline and analyze whether each method retrieves images with similar table states or is biased toward similar viewpoints.
**Input:**
- query images: test
- retrieval database: train
- split from `partition.csv`

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


### **Experiment 2:  Deep Feature Retrieval**

- **Goal:** Evaluate whether deep visual features extracted from a pretrained CNN provide a better retrieval representation than direct pixel-level similarity.

**Input:**
- query images: test
- retrieval database: train
- split from `partition.csv`

**Methods:**
- Pretrained ResNet embeddings + cosine similarity
    - Use a pretrained ResNet model as a feature extractor.
    - Remove the final classification layer.
    - Extract one embedding vector per image.
    - Compare query embeddings against retrieval-pool embeddings using cosine similarity.

- Task-specific CNN embeddings + cosine similarity
    - Use the CNN trained in Task 2.
    - Remove the final prediction layer.
    - Extract embeddings from the last feature layer.
    - Compare embeddings using cosine similarity.

**Output:**
For each query image:
- Query image
- Top-5 retrieved images using ResNet embeddings
- Cosine similarity scores

**Expected:**
- ResNet may retrieve images based on global visual semantics and viewpoint.
- The task-specific CNN may retrieve images with more relevant billiard-table content because its features are learned from the target domain rather than from generic ImageNet categories.
- However, because Task 2 is trained for counting, its embeddings may not fully capture spatial ball configuration.




**Comparison Objective:**
- Compare retrieval quality between:
  - MSE
  - SSIM
  - ResNet embeddings
  - Task-specific CNN embeddings