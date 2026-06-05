# Clinical Analytics Engine: Algorithmic Fairness & Resource Governance

An end-to-end healthcare analytics platform designed to optimize clinical predictive workflows, audit algorithmic fairness, and uncover prescription-to-hospitalization resource drivers.

This repository leverages **XGBoost** paired with **Fairlearn** to implement a bias-mitigated readmission safety net, and the **Apriori Algorithm** alongside **Parametric Inference (ANOVA + Tukey HSD)** to map out diabetes medication co-prescriptions and their direct impacts on hospital Length of Stay (LOS).

---

## Project Architecture

The system treats hospital data through two decoupled, complementary analytical pipelines:
1. **The Governance Lens:** Mitigates systemic bias by training machine learning models blindly (withholding demographic traits) and applying cost-sensitive learning to optimize equitable patient care boundaries across demographic slices.
2. **The Optimization Lens:** Explores patient transactional medication profiles using market basket analysis and statistical inference to map how clinical decision-making affects macro-level hospital capacity.

---

## Pipeline 1: Readmission Classification & Fairness Audit

### 1. Exploratory Data Analysis & Sample Auditing
* **Operational Imbalance:** Resolves severe class scarcity; acute early readmissions ($<30$ days) represent a low baseline prevalence of **~9.59%**.
* **Data Sparsity Mitigation:** Tracks and drops high-variance unclassified demographics (e.g., 'Other' race) and trims medical profile outliers to focus exclusively on patients presenting stable comorbidity baselines (between 3 and 9 diagnoses).
* **Statistical Convergence:** Proves via frequentist tracking that at peak clinical complexity (9 diagnoses), the resource distribution median variance collapses across all racial groups ($\Delta \le 4.0$ lab procedures). Conversely, it highlights that at lower tiers (3 diagnoses), severe data sparsity ($n=16$ for Asian patients) induces visual chart noise rather than systemic bias.

### 2. Parametric Baseline (OLS & Type-II ANOVA)
Fitting an Ordinary Least Squares ($OLS$) interaction model establishes institutional integrity:
* **Clinical Driver ($C(\text{number\_diagnoses})$):** Strongly dictates procedure volumes ($F = 249.74, p \approx 0$).
* **Demographic Baseline ($C(\text{race})$):** Flags statistically significant independent variance ($F = 28.61, p = 1.77 \times 10^{-18}$), verifying the prerequisite for algorithmic auditing.
* **Interaction Shift ($C(\text{race}):C(\text{number\_diagnoses})$):** Proves non-significant ($p = 0.089 > 0.05$), mathematically confirming that the baseline resource gap remains parallel and does not compound or worsen as a patient becomes sicker.

### 3. Cost-Sensitive XGBoost Optimization & Fairlearn Audit
* **Feature Engineering:** Features are cast directly to categorical channels to harness native tree-splitting optimizations and avoid one-hot dimensionality explosions.
* **Fairlearn Matrix Evaluation:** Models are trained blindly without sensitive features. Prediction arrays are audited across progressive class weight multipliers ($X1$ to $X10$) to combat minority class under-detection.
* **Equity Milestone:** The chosen **$X7$ penalty weight** with a post-hoc probability threshold of **$\ge 0.45$** successfully compresses the False Negative Rate Disparity ($\Delta FNR$) to just **$5.88\%$**, falling safely under the $10\%$ international algorithmic compliance threshold.

---

## Pipeline 2: Medication Regimen Analytics & Resource Governance

This pipeline utilizes market basket analysis to isolate frequent medication co-prescriptions, treatments, and drug combinations, running them through parametric evaluation frameworks to pinpoint how polypharmacy profiles directly drive variations in hospital Length of Stay (LOS).

### 1. Ingestion & Transaction Encoding
Parses historical patient records to construct active medication vectors. By processing each patient's active script list as a distinct transaction matrix, drug regimens are normalized into generic categories to eliminate redundant chemical channels.

### 2. Market Basket Analysis (Apriori Engine)
The pipeline utilizes the **Apriori Algorithm** to mine frequent itemsets and structural co-prescriptions. Relationships are filtered and verified through three main mathematical thresholds: Support, Confidence, and Lift. This maps hidden multi-drug dependencies and visualizes the strength of drug-to-drug interactions across the clinical enterprise.

### 3. Parametric Length of Stay (LOS) Modeling
* **One-Way ANOVA:** Executes global hypothesis testing to mathematically prove whether variances in a patient's hospitalization days (Length of Stay) are significantly driven by specific co-prescription regimens rather than arbitrary operational fluctuations.
* **Post-Hoc Tukey HSD (Honestly Significant Difference):** If the global ANOVA rejects the null hypothesis, a pairwise Tukey HSD test is deployed. This robustly controls the family-wise error rate across multiple comparisons, pinpointing the precise medication pairs that significantly extend or minimize hospitalization footprints.

---

## Production Deployment Standards

The serialized artifacts generated in this project are optimized for seamless integration within an Electronic Health Record (EHR) discharge workflow:
1. **Core Safeguard Engine:** Deploy `final_model.pkl` using an operational minority class weight penalty of **7** and a calibrated decision boundary threshold of **$\ge 0.45$**.
2. **Staff Alert Governance:** With a controlled selection rate rolling between $30.6\%$ and $36.7\%$, the system prevents **"Alert Fatigue"** among clinical staff, filtering out noise while securing a defensive safety net for critical early readmissions.

---

## Installation & Execution

### Prerequisites
* Python 3.10 or higher
* Virtual Environment activated (`.venv`)

### 1. Setup Environment
```bash
git clone [https://github.com/dilyshuynh3864/Hospital-Management.git](https://github.com/dilyshuynh3864/Hospital-Management.git)
cd Hospital-Management
pip install -r requirements.txt
```
### 2. Execute Analytics Pipelines
Open the notebooks locally to step through the code execution loops:
``` bash
jupyter notebook
```
* Navigate to `notebooks/01_readmission_fairness_audit.ipynb` for the machine learning auditing suite.
* Navigate to `notebooks/02_medication_market_basket_los.ipynb` for the market basket and statistical modeling suite.

## Repository Structure
```text
Hospital-Management/
├── data/
│   ├── processed.csv               # Unified historical clinical tracker
│   └── fairlearn_audit_summary.csv # Progressive fairness validation records
├── notebooks/
│   ├── readmission_fairness_audit.ipynb   # XGBoost + Fairlearn Pipeline
│   └── medication_market_basket_los.ipynb  # Apriori + ANOVA/Tukey Pipeline
│── cat_mapping.pkl             # Encoded feature matrix categories
│── label_encoder.pkl           # Target variable string-to-numeric encoder
│── final_model.pkl             # Production cost-sensitive XGBoost model
├── requirements.txt                # Unified project dependencies
└── README.md                       # Comprehensive documentation
```
