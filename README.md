# CSCI566 Project: Diabetic Retinopathy Detection

## Data Sources

| Dataset | Size | Link |
|---------|------|------|
| EyePACS | 88,702 images | [Kaggle Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection) |

## Related Work

- **Google's DR Detection (2016)**: Achieved **AUC 0.97** for diabetic retinopathy detection  
  [JAMA Publication](https://jamanetwork.com/journals/jama/fullarticle/2588763)

---

## 1. Motivation

Diabetic retinopathy (DR) is the **leading cause of preventable blindness** among working-age adults worldwide. According to the International Diabetes Federation:

- ~**537 million** adults have diabetes globally
- ~**1 in 3** will develop some form of DR
- Early detection can prevent up to **95%** of vision loss

Yet many patients lack access to timely eye exams due to specialist shortages.

### Clinical Background

Screening for DR relies on analyzing **color fundus photographs** of the retina to identify lesions such as:

- **Microaneurysms** — tiny red dots
- **Hemorrhages** — bleeding in the retina
- **Exudates** — yellow lipid deposits

These findings can be subtle and easily missed, especially across large-scale screening programs. Our project focuses on **automated DR severity classification** as a decision-support tool for clinicians. While the system does not replace clinical diagnosis, it aims to flag high-risk cases for priority review, potentially supporting earlier intervention.

---

## 2. Proposed Approach

Our project proposes an assistive diabetic retinopathy classification pipeline using deep learning and fundus images from the Kaggle EyePACS and APTOS 2019 datasets.

### Pipeline Overview

#### Step 1: Fundus Image Preprocessing
- Load color fundus images from the dataset
- Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance lesion visibility
- Resize and normalize images for model input

#### Step 2: Classification Model Development
- Train a CNN (**EfficientNet** or **ResNet**) to classify DR severity into 5 levels:
  - 0 — No DR
  - 1 — Mild
  - 2 — Moderate
  - 3 — Severe
  - 4 — Proliferative DR
- Use clinician-provided severity labels from the dataset as ground truth
- Apply **class-weighted loss functions** to handle imbalanced severity distributions

### Design Goals

The focus of our project is on building a **reliable classification system** that accurately grades DR severity from fundus images as a decision-support tool.

---

## 3. Expected Outcomes

By the end of the project, we expect to:

1. **Data Preprocessing**: Successfully preprocess raw fundus images into model-ready inputs with appropriate normalization and augmentation
2. **Classification Model**: Build and evaluate a working DR classification model that automatically grades disease severity
3. **Performance Evaluation**: Assess model performance using standard metrics:
   - **AUC** (Area Under ROC Curve)
   - **Sensitivity** and **Specificity**
   - **Quadratic Weighted Kappa** — the standard metric for ordinal classification

The results aim to demonstrate the feasibility of deep learning-based assistive screening for diabetic retinopathy.

---

## 4. Potential Impact

Our project would contribute to the broader effort of improving computer-aided diabetic retinopathy screening by investigating how automated classification models can support clinicians in analyzing fundus images more efficiently and consistently.

### Clinical Applications

- **Triage high-risk cases** — identify patients requiring urgent specialist attention
- **Enable earlier referral** — support timely intervention before vision loss
- **Improve screening consistency** — reduce variability in grading across programs

Even modest improvements in screening accuracy and consistency can have meaningful clinical value, particularly in **resource-limited settings** where ophthalmologist access is scarce.
