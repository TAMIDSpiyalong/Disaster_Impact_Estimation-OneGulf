# Disaster Impact Estimation - OneGulf (working in progress)

## Introduction

This repository is built around the problem of **disaster-related damage estimation** and **impact classification** using social media (Twitter) text data. Specifically, it focuses on **multi-label classification** for various FEMA-assistance-related targets, including but not limited to **insurance eligibility, property damage, and displacement support**. The core idea is to predict these FEMA-relevant labels from tweet content using **deep neural network architectures**, particularly transformer-based model - **BERT**.

The experiments simulate real-world deployment scenarios for rapid damage assessment following major storms such as **Harvey, Beryl, Imelda, Hanna and Laura**. The evaluation includes **in-domain**, **cross-domain**, and **ensemble** settings. Model evaluation is based on k-fold validation, multi-label classification metrics, and in some cases, region-specific post-analysis to assess geographic sensitivity.

---

## Data Collection

The dataset includes Twitter messages related to individual storm events. Each storm-specific dataset contains:

* Raw tweet text https://data.griidc.org/data/O1.x158.000:0003 
* FEMA-target labels https://data.griidc.org/data/O1.x158.000:0004
* Optional geolocation data (e.g., zip code centroids, bounding boxes)
* Disaster metadata

  ### Bounding Box Filtering

To avoid noisy or irrelevant geospatial data, bounding boxes that exceed a predefined area threshold are discarded. Bounding box area is calculated using the **Haversine formula** to measure surface distance across coordinates. A bounding box is removed if:

```python
get_box_area(lat1, lon1, lat2, lon2) > MAX_ALLOWED_AREA
```

This ensures high-quality, localized disaster signal detection.

### Zip Code Classification

To associate Twitter activity with FEMA targets, zip codes are assigned **positive** or **negative** labels for each FEMA category:

* **Positive Zip**: Contains at least one positive FEMA label with value 1 (e.g., roof damage).
* **Negative Zip**: No FEMA records (e.g., flood damage) within the zip code.

These are later paired with aggregated tweet text for model training.

Each tweet (from a unique zipcode) may be labeled with multiple targets, such as:

* `homeOwnersInsurance`
* `floodInsurance`
* `destroyed`
* `roofDamage`
* `tsaEligible`
* `tsaCheckedIn`
* `rentalAssistanceEligible`
* `repairAssistanceEligible`
* `replacementAssistanceEligible`
* `personalPropertyEligible`

Datasets are preprocessed with tokenization (BERT tokenizer), truncation, and padding. Additionally, bounding box information is used for geospatial filtering and domain adaptation.

---

## Vectorization and Model Architectures

### Tokenization

Tweets are tokenized using the **BERT tokenizer (`bert-base-uncased`)**, which performs:

* WordPiece tokenization
* Padding/truncation to fixed length (e.g., 256 or 512)
* Addition of special tokens (\[CLS], \[SEP])

This allows the input to align with pretrained transformer encoders.

### BERT-Based Deep Multi-Head Classifier

Implemented in `BERTMultiDeepHeadClassifier`, this architecture features:

* **Shared BERT encoder** for all targets
* **Multiple heads**, one per FEMA label
* Each head is a deep MLP with 512 → 128 → 1 units and sigmoid activation

Loss is calculated using **binary cross-entropy (BCE)** across all heads. Training includes:

* 5-Fold Cross Validation
* Adam optimizer with learning rate 1e-5
* Early stopping with patience of 15
* Learning rate scheduling
* Optional mixed precision training

### Longformer (Alternative Model)

For long-context modeling (especially when tweet batches are large per zip), **Longformer** is explored using:

* Efficient sparse self-attention mechanism
* Supports sequences beyond 512 tokens
* Same multi-head architecture over pooled output

Longformer uses the same tokenizer and classification logic but benefits from scalability over large token windows.

---


## Modeling

Multiple architectures were explored, including:

* **BERT-based Deep Multi-Head Classifier**: A custom model with one classification head per FEMA target.
* **Ensemble and Cross-Domain Models**: Evaluated the generalizability of models trained on one storm to others.

Model saving was performed per fold, allowing reuse and inference across experiments.

![disaster_imapct_estimation_bert_model_schematic_diagram](https://github.com/user-attachments/assets/989f914d-3d8a-4a86-b379-1bc14392cafd)

---

### **Master Table: Model Performance Comparison (BERT Base with Multi-Head)**
| **Train Dataset**       | **Eval Dataset** | **Token Length** | **Avg Accuracy** | **Avg F1** | **Avg Precision** | **Avg Recall** | **Notes**                          |
|--------------------------|------------------|------------------|------------------|------------|-------------------|----------------|------------------------------------|
| Harvey                   | Harvey           | 256              | 0.9877           | 0.9906     | 0.9928            | 0.9885         | High performance (in-domain)       |
| Harvey                   | Harvey           | 512              | 0.9918           | 0.9936     | 0.9946            | 0.9927         | Slight improvement with longer tokens |
| Beryl                    | Beryl            | 256              | 0.9830           | 0.9005     | 0.8935            | 0.9079         | Low F1 due to `roofDamage` (0 support) |
| Beryl                    | Beryl            | 512              | 0.9841           | 0.9005     | 0.8923            | 0.9091         | Similar to 256-token version       |
| Imelda                   | Imelda           | 256              | 0.9616           | 0.7678     | 0.7547            | 0.7901         | Low F1 due to `tsaEligible/CheckedIn` (0 support) |
| Imelda                   | Imelda           | 512              | 0.9545           | 0.7579     | 0.7149            | 0.8119         | Recall improves but precision drops |
| Beryl+Imelda             | Harvey           | 512              | 0.6373           | 0.5600     | 0.5017            | 0.6625         | Poor generalization to Harvey      |
| Harvey+Imelda            | Beryl            | 512              | 0.6523           | 0.5651     | 0.7089            | 0.4871         | Low recall (underfitting)          |
| Harvey+Beryl             | Imelda           | 512              | 0.5584           | 0.4510     | 0.4585            | 0.5944         | Struggles with Imelda's distribution |
| Harvey+Beryl+Imelda      | Harvey           | 256              | 0.9456           | 0.9436     | 0.9336            | 0.9689         | Solid cross-dataset performance    |
| Harvey+Beryl+Imelda      | Beryl            | 256              | 0.9261           | 0.8277     | 0.7979            | 0.9006         | Good recall, lower precision       |
| Harvey+Beryl+Imelda      | Imelda           | 256              | 0.8335           | 0.6540     | 0.6829            | 0.6399         | Struggles with Imelda              |
| Harvey+Beryl+Imelda      | Combined         | 256              | 0.9163           | 0.9061     | 0.8751            | 0.9561         | Best overall generalization        |

---

### **Key Observations**
- **In-Domain vs. Cross-Dataset**: Models excel when trained/tested on the same disaster (e.g., Harvey→Harvey) but struggle with cross-dataset evaluation.  
- **Token Length**: 512 tokens slightly outperform 256 in-domain but offer no clear advantage for generalization.  
- **Mixed Training Data**: Combining Harvey+Beryl+Imelda improves cross-dataset performance (e.g., 0.92 accuracy on combined evaluation).  
- **Class Imbalance**: Rare classes (`destroyed`, `replacementAssistanceEligible`) suffer in cross-dataset tests due to overfitting.  
- **Zero-Support Issues**: Targets like `roofDamage` fail when training data has no examples (e.g., Beryl training).  

---

## Inference

Inference pipelines were developed to predict on available social media datasets:

* Load saved model checkpoints per fold
* Tokenize new tweet batches
* Run predictions and threshold output probabilities
* Output results as CSVs containing predictions per FEMA target

---

## Visualization

Given below are some of the prediction targets that have been predicted using the above mentioned models.

![flooddamage](https://github.com/TAMIDSpiyalong/Disaster_Impact_Estimation-OneGulf/blob/main/Harvey%20Flood%20Dmg.jpg)
![roofamage](https://github.com/TAMIDSpiyalong/Disaster_Impact_Estimation-OneGulf/blob/main/Harvey%20Roof%20Dmg.jpg)

The ground-truth FEMA record visualization displays the total number of households for each category. 
![flooddamage](https://github.com/TAMIDSpiyalong/Disaster_Impact_Estimation-OneGulf/blob/main/floodDamage.jpg)
![roofamage](https://github.com/TAMIDSpiyalong/Disaster_Impact_Estimation-OneGulf/blob/main/roofDamage.jpg)

---

## Conclusion

Transformer-based models trained on social media data can capture significant damage signals post-disaster, achieving up to **70% agreement with FEMA labels**. Models are capable of generalizing across storm domains and can be used for **real-time estimation** in under-surveyed regions. Ensemble techniques and domain adaptation further improve reliability, enabling scalable and ethical deployment of AI for disaster relief support.

