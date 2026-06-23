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
│   ├── setup.py          # device setting
├── exp1_raw_baseline.ipynb
├── exp2_deep_feature_retrieval.ipynb
```


## **Experiments**

#### **Experiments Overview**
| Experiment  | Brief Description | Status  | Results | Observations | 
|------------|-------------------|--------|---------------------------|--------|
| E1 | Raw Image Retrieval Baseline | [x] Done  | [exp1_raw_baseline](experiments/exp1_raw_baseline.ipynb) | Based on the professor's baseline notebook |
| E2 | Deep Feature Retrieval: ResNet, ViT, Task 2 CNN + Cosine Similarity | [-] Progress | - | - |
| E3 | Multi-view Contrastive Retrieval | [] | - | - |
| E4 | Top-view Normalized Retrieval | [] | - | - |


----

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

**Output:**
For each query image:
- Query image
- Top-5 retrieved images with MSE
- Top-5 retrieved images with SSIM

**Expected:** 
- MSE and SSIM will probably retrieve images with similar camera pose / similar colors instead of same or similar table state



----
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

- Pretrained ViT embeddings + cosine similarity
  - Use a pretrained Vision Transformer as an alternative generic feature extractor.
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
- Top-5 retrieved images using ViT embeddings
- Top-5 retrieved images using Task2 CNN embeddings
- Cosine similarity scores

**Expected:**
- ResNet may retrieve images based on global visual semantics and viewpoint.
- ViT may capture more global spatial relationships than ResNet due to its attention-based architecture.
- However, since it is also pretrained on generic image data, it may still suffer from viewpoint bias and may not explicitly encode ball configurations.
- The task-specific CNN may retrieve images with more relevant billiard-table content because its features are learned from the target domain rather than from generic ImageNet categories.
- However, because Task 2 is trained for counting, its embeddings may not fully capture spatial ball configuration.

**Comparison Objective:**
- Compare retrieval quality between:
  - MSE
  - SSIM
  - ResNet embeddings
  - ViT embeddings
  - Task-specific CNN embeddings


-----
### **Experiment 3: Multi-view Representation Learning**

- **Goal:** Learn embeddings where images of the same table state are close together, even when captured from different viewpoints.

**Motivation:**
- The dataset intentionally contains multiple views of the same table state.
- Previous methods may retrieve images with similar camera pose instead of similar ball configurations.
- This experiment directly addresses viewpoint bias by using multi-view image groups during training.

**Input:**
- query images: test
- retrieval database: train
- split from `partition.csv`
- multi-view groups inferred from filenames, e.g. `25.png`, `25a.png`, `25f.png`

**Methods:**

1. **State Classification Proxy Task**
   - Group images by base table-state ID.
   - Images from the same multi-view group share the same class.
   - Train a CNN to classify the table-state ID.
   - Remove the final classification layer.
   - Use the learned embedding for retrieval.

2. **Contrastive Learning**
   - Positive pairs: images from the same table-state group.
   - Negative pairs: images from different table-state groups.
   - Train the embedding space so positives are close and negatives are far.

3. **Triplet Learning**
   - Anchor: one image from a table-state group.
   - Positive: another viewpoint of the same table state.
   - Negative: image from a different table state.
   - Train the model so the anchor is closer to the positive than to the negative.

**Output:**
For each query image:
- Query image
- Top-5 retrieved images using learned multi-view embeddings
- Similarity scores

**Expected:**
- Different viewpoints of the same table state should become closer in embedding space.
- Retrieval should become less biased toward camera pose.
- Contrastive/triplet learning should be more directly aligned with retrieval than classification.
- Performance may be limited by the number and quality of multi-view groups.

**Comparison Objective:**
Compare against:
- MSE
- SSIM
- ResNet embeddings
- ViT embeddings
- Task 2 CNN embeddings



### **Experiment 3.1: Combination of embeedings**

Usar embeddings da imagem + bolas + ...


----
### **Experiment 4: Top-view Normalized Retrieval**

- **Goal:** Remove viewpoint variation by representing each image in a common table coordinate system before retrieval.

**Motivation:**
- The final retrieval objective is to compare billiard-table states rather than camera viewpoints.
- The professor explicitly suggested incorporating a top-view representation into the retrieval pipeline.
- Instead of learning viewpoint invariance, this experiment attempts to eliminate viewpoint differences geometrically.

**Input:**
- query images: test
- retrieval database: train
- split from `partition.csv`

**Method:**

For each image:

```text
Image
→ Table Detection
→ Perspective Correction
→ Top-view Table Representation
```

The resulting top-view images are then used as the retrieval representation.

Possible retrieval approaches:

- Pixel similarity on top-view images (MSE / SSIM)
- Deep embeddings extracted from top-view images
- Ball-position representations extracted from the top-view image

**Output:**
For each query image:

- Query image
- Top-5 retrieved images
- Similarity scores

**Expected:**

- Images representing the same table state should become more similar regardless of the original camera viewpoint.
- Retrieval should become less sensitive to perspective distortion.
- Ball positions should have greater influence on retrieval than camera pose.
- Performance will depend on the quality of the table detection and perspective correction stage.

**Comparison Objective:**
Compare:

- Raw-image retrieval (E1)
- Deep feature retrieval (E2)
- Multi-view representation learning (E3)
- Top-view normalized retrieval (E4)
