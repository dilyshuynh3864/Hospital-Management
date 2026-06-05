# Clinical Analytics Engine: Algorithmic Fairness & Resource Governance

An end-to-end, dual-pipeline healthcare analytics platform designed to optimize clinical predictive workflows, audit algorithmic fairness, and uncover prescription-to-hospitalization resource drivers. 

This repository leverages **XGBoost** paired with **Fairlearn** to implement a bias-mitigated readmission safety net, and the **Apriori Algorithm** alongside **Parametric Inference (ANOVA + Tukey HSD)** to map out diabetes medication co-prescriptions and their direct impacts on hospital Length of Stay (LOS).

---

## 📋 Table of Contents
- [Project Architecture](#-project-architecture)
- [Pipeline 1: Readmission Classification & Fairness Audit](#-pipeline-1-readmission-classification--fairness-audit)
- [Pipeline 2: Medication Regimen Analytics & Resource Governance](#-pipeline-2-medication-regimen-analytics--resource-governance)
- [🚀 Production Deployment Standards](#-production-deployment-standards)
- [📦 Installation & Execution](#-installation--execution)
- [📁 Repository Structure](#-repository-structure)

---

## ⚙️ Project Architecture

The system treats hospital data through two decoupled, complementary analytical pipelines:

1. **The Governance Lens:** Mitigates systemic bias by training machine learning models blindly (withholding demographic traits) and applying cost-sensitive learning to optimize equitable patient care boundaries across demographic slices.
2. **The Optimization Lens:** Explores patient transactional medication profiles using market basket analysis and statistical inference to map how clinical decision-making affects macro-level hospital capacity.

---

## 🔍 Pipeline 1: Readmission Classification & Fairness Audit

### 1. Exploratory Data Analysis & Sample Auditing
* **Operational Imbalance:** Resolves severe class scarcity; acute early readmissions ($<30$ days) represent a low baseline prevalence of **~9.59%**.
* **Data Sparsity Mitigation:** Tracks and drops high-variance unclassified demographics (e.g., 'Other' race) and trims medical profile outliers to focus exclusively on patients presenting stable comorbidity baselines (between 3 and 9 diagnoses).
* **Statistical Convergence:** Proves via frequentist tracking that at peak clinical complexity (9 diagnoses), the resource distribution median variance collapses across all racial groups ($\Delta \le 4.0$ lab procedures). Conversely, it highlights that at lower tiers (3 diagnoses), severe data sparsity ($n=16$ for Asian patients) induces visual chart noise rather than systemic bias.

### 2. Parametric Baseline (OLS & Type-II ANOVA)
Fitting an Ordinary Least Squares ($OLS$) interaction model establishes institutional integrity:
* **Clinical Driver ($C(\text{number\_diagnoses})$):** Strongly dictates procedure volumes ($F = 249.74, p \approx 0$).
* **Demographic Baseline ($C(\text{race})$):** Flags statistically significant independent variance ($F = 28.61, p = 1.77 \times 10^{-18}$), verifying the prerequisite for algorithmic auditing.
* **Interaction Shift ($C(\text{race}):C(\text{number\_diagnoses})$):** Proves non-significant ($p = 0.089 > 0.05$), mathematically confirming that the baseline resource gap remains parallel and does not compound or worsen as a patient becomes sicker.

### 3. Cost-Sensitive XGBoost Optimization
* **Feature Engineering:** Features are cast directly to categorical channels to harness native tree-splitting optimizations and avoid one-hot dimensionality explosions.
* **Fairlearn Matrix Evaluation:** Models are trained blindly without sensitive features. Prediction arrays are audited across progressive class weight multipliers ($X1$ to $X10$) to combat minority class under-detection.

```text
=== Final Calibrated Fairness Audit Log (Lens 1) ===
Gender Cohort    Accuracy    Precision    Selection Rate     False Negative Rate (FNR)
Female (0)       0.624104    0.121212     0.367220           0.544402
Male (1)         0.674334    0.121697     0.306283           0.603175


## 💊 Pipeline 2: Medication Regimen Analytics & Resource Governance

This pipeline explores patient transactional medication profiles using market basket analysis and statistical inference to map how clinical decision-making affects macro-level hospital capacity.

### 1. Ingestion & Transaction Encoding
Parses historical patient records to construct active medication vectors. By processing each patient's active script list as a distinct transaction matrix, drug regimens are normalized into generic categories to eliminate redundant chemical channels.

### 2. Market Basket Analysis (Apriori Engine)
The pipeline utilizes the **Apriori Algorithm** to mine frequent itemsets and structural co-prescriptions. Relationships are filtered and verified through three main mathematical thresholds: Support, Confidence, and Lift. This maps hidden multi-drug dependencies and visualizes the strength of drug-to-drug interactions across the clinical enterprise.

### 3. Parametric Length of Stay (LOS) Modeling
* **One-Way ANOVA:** Executes global hypothesis testing to mathematically prove whether variances in a patient's hospitalization days (Length of Stay) are significantly driven by specific co-prescription regimens rather than arbitrary operational fluctuations.
* **Post-Hoc Tukey HSD (Honestly Significant Difference):** If the global ANOVA rejects the null hypothesis, a pairwise Tukey HSD test is deployed. This robustly controls the family-wise error rate across multiple comparisons, pinpointing the precise medication pairs that significantly extend or minimize hospitalization footprints.
