# Disaster Impact Estimation - OneGulf

## Introduction

This repository is built around the problem of **disaster-related damage estimation** and **impact classification** using social media (Twitter) text data. Specifically, it focuses on **multi-label classification** for various FEMA-assistance-related targets, including but not limited to **insurance eligibility, property damage, and displacement support**. The core idea is to predict these FEMA-relevant labels from tweet content using **deep neural network architectures**, particularly transformer-based model - **BERT**.

The experiments simulate real-world deployment scenarios for rapid damage assessment following major storms such as **Harvey, Beryl, Imelda, Hanna and Laura**. The evaluation includes **in-domain**, **cross-domain**, and **ensemble** settings. Model evaluation is based on k-fold validation, multi-label classification metrics, and in some cases, region-specific post-analysis to assess geographic sensitivity.

---

## Data Collection

The dataset includes Twitter messages related to individual storm events. Each storm-specific dataset contains:

* Raw tweet text
* FEMA-target labels (multi-label binary annotations)
* Optional geolocation data (e.g., zip code centroids, bounding boxes)
* Disaster metadata

Each tweet may be labeled with multiple targets such as:

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

## Modeling

Multiple architectures were explored, including:

* **BERT-based Deep Multi-Head Classifier**: A custom model with one linear classification head per FEMA target, preceded by a shared BERT encoder and a dropout layer.
* **Ensemble and Cross-Domain Models**: Evaluated the generalizability of models trained on one storm to others.

Training involved:

* 5-fold cross-validation
* AdamW optimizer
* Binary cross-entropy loss
* Early stopping
* Gradient accumulation
* Optional mixed-precision training

Model saving was performed per fold, allowing reuse and inference across experiments.

---

## Inference

Inference pipelines were developed to:

* Load saved model checkpoints per fold
* Tokenize new tweet batches
* Run predictions and threshold output probabilities
* Output results as CSVs containing predictions per FEMA target

Experiments also included **"negative-only" inference** on storms like **Laura**, to simulate deployment where labels are unavailable but real-time classification is needed.

---

## Evaluation

Each model's predictions were evaluated using:

* **F1 Score (macro, micro, per label)**
* **Precision & Recall**
* **Multi-label Confusion Matrices**
* **Cross-domain transfer accuracy**

Visualization: 
Given below are some of the few predicted targets that have been predicted using the above mentioned models.
![tsaEligible](https://github.com/user-attachments/assets/dd43cc98-6788-4da0-a8da-cd2cc807a2e8)
![tsaCheckedIn](https://github.com/user-attachments/assets/482d113a-8bcb-4657-9f68-91f0ea352a26)
![roofDamage](https://github.com/user-attachments/assets/17aec4ee-7713-4035-9cb0-57f1c159b87c)
![rentalAssistanceEligible](https://github.com/user-attachments/assets/0fa0015f-b292-4c0a-b3e4-28e6224d0311)
![homeDamage](https://github.com/user-attachments/assets/6c83cb3e-0c46-4c5a-b10a-08d9501109f0)


---

## Conclusion

Transformer-based models trained on social media data can capture significant damage signals post-disaster, achieving up to **70% agreement with FEMA labels**. Models are capable of generalizing across storm domains and can be used for **real-time estimation** in under-surveyed regions. Ensemble techniques and domain adaptation further improve reliability, enabling scalable and ethical deployment of AI for disaster relief support.

