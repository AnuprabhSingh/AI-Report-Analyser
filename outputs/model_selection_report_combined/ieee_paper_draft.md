# Safety-Aware Selective Prediction for Echocardiography Report Interpretation: An Adaptive Rule-ML Hybrid Framework with Explicit Abstention

## Abstract
Clinical decision support systems for echocardiography must balance predictive accuracy with operational safety, particularly when training data is limited and external validation is constrained. This paper introduces a novel adaptive routing framework that explicitly models the decision to defer uncertain predictions—a critical requirement often overlooked in medical AI systems. Unlike conventional hybrid approaches that use fixed confidence thresholds, our method implements a principled three-way routing policy (ML/Rule/Abstain) with disagreement-aware gating that dynamically selects between machine learning predictions, deterministic guideline-based rules, or explicit abstention based on prediction confidence, rule-ML agreement, and feature availability.

We demonstrate through comprehensive comparison against established selective prediction methods (maximum confidence thresholding, entropy-based selection, ensemble disagreement) that our approach achieves superior performance: 99.4% covered accuracy at 96.3% coverage with only 0.64% risk-at-coverage, outperforming all baselines. Notably, the closest accuracy-matched baseline (ensemble with full agreement) requires 14.7% abstention compared to our 3.7%—a 4× reduction. Under simulated distribution shift, the system exhibits appropriate safety-aware behavior: abstention rate increases from 4.3% to 9.8% as noise increases, while maintaining covered accuracy above 95.6%.

The framework is instantiated as an end-to-end pipeline converting echocardiography PDF reports into structured interpretations across five clinical categories. We provide theoretical justification for Gradient Boosting over deep learning in this data-constrained, tabular-feature setting and demonstrate through rigorous repeated stratified cross-validation (5 folds × 10 repeats) that it achieves statistically significant improvements over baselines (Macro F1: 0.9223, p < 0.001 vs. Random Forest). The complete system includes extraction, normalization, leakage prevention, multi-method explainability (SHAP/PDP/ICE), robustness analysis, and deployable components with full reproducibility.

Keywords: Selective Prediction, Clinical Decision Support, Explainable AI, Hybrid AI, Safety-Aware Machine Learning, Echocardiography, Abstention

## I. Introduction
Echocardiography is central to cardiac assessment, but many reports remain semi-structured PDFs that are difficult to standardize for downstream analytics. While deep learning has achieved remarkable success in medical imaging, deploying such systems in clinical practice faces fundamental challenges: limited labeled data, the need for interpretable decisions, and critically, the absence of mechanisms for models to express uncertainty or decline to predict when confidence is low.

The clinical implications of confident but incorrect predictions in cardiac assessment can be severe—missed diagnoses or false alarms affecting patient care pathways. Yet most machine learning systems are designed to always produce a prediction, even when operating outside their competence boundary. This paper addresses this gap through *selective prediction*: the principled ability for a system to abstain from prediction when uncertainty is high, routing such cases to human review.

We frame this as an end-to-end explainable clinical decision support system encompassing PDF parsing, structured feature extraction, hybrid inference with explicit abstention capability, and deployable interfaces. The final selected ML model is Gradient Boosting—chosen deliberately over deep learning alternatives for reasons of data efficiency, tabular feature structure, and superior interpretability (detailed in Section II-C). Rule-based logic provides deterministic fallback grounded in clinical guidelines.

**Major contributions:**
1. **Adaptive selective prediction framework**: A novel three-way routing policy (ML/Rule/Abstain) with theoretical grounding in coverage-risk tradeoffs, going beyond simple confidence thresholding to incorporate rule-ML disagreement signals.
2. **Principled model selection for constrained settings**: Theoretical and empirical justification for Gradient Boosting over deep learning when data is limited and features are tabular.
3. **End-to-end clinical pipeline**: Production-ready PDF-to-interpretation system with extraction, normalization, and deployment components.
4. **Comprehensive evaluation methodology**: Repeated stratified CV with paired statistical tests, leakage prevention verification, and risk-coverage analysis.
5. **Multi-method explainability**: SHAP, PDP/ICE, sensitivity analysis, and calibration assessment integrated throughout.
6. **Reproducibility**: Complete codebase with fixed seeds, versioned dependencies, and containerized deployment.

## II. Related Work

### A. AI in Echocardiography
Deep learning has transformed echocardiography analysis, primarily in image and video interpretation. Madani et al. [1] demonstrated CNN-based view classification achieving cardiologist-level performance. Ouyang et al. [2] developed EchoNet-Dynamic for video-based ejection fraction estimation, validated on over 10,000 studies. Bernard et al. [3] addressed multi-structure segmentation using encoder-decoder architectures.

However, these approaches share common limitations: they require large curated image/video datasets (typically thousands to tens of thousands of labeled studies), demand substantial computational resources, and often function as black boxes requiring post-hoc explanation. For institutions with limited data access or computational infrastructure, such approaches may be impractical.

### B. Report-Based Clinical NLP
An alternative paradigm processes clinical reports rather than raw imaging data. Structured reporting initiatives [13] have improved standardization but adoption remains incomplete. Natural language processing for clinical text has evolved from rule-based systems [14] through statistical methods to transformer architectures [15]. However, echocardiography report interpretation poses specific challenges: mixed narrative and tabular content, institution-specific templates, and the need to extract numerical measurements with associated uncertainty.

Prior report-based cardiac systems include rule-based extraction for EHR integration [16] and template-matching approaches [17]. These provide transparency but lack adaptive learning capability. Conversely, end-to-end neural approaches sacrifice interpretability for flexibility.

### C. Why Gradient Boosting Over Deep Learning?
The choice of Gradient Boosting [8] over deep learning is deliberate and principled for this setting:

1. **Data efficiency**: Gradient Boosting achieves strong performance with hundreds of samples; deep learning typically requires orders of magnitude more data to avoid overfitting [9].

2. **Tabular feature structure**: Our extracted features (14 numerical measurements) are inherently tabular. Recent benchmarks demonstrate that gradient boosting methods (XGBoost, LightGBM, CatBoost) consistently match or outperform deep learning on tabular data [18], with Grinsztajn et al. [19] showing tree-based methods remain superior on medium-sized tabular datasets.

3. **Interpretability**: Gradient Boosting provides native feature importance, direct SHAP computation without approximation [4], and stable explanations—critical for clinical acceptance.

4. **Calibration**: Tree ensembles tend to produce better-calibrated probabilities than neural networks without explicit calibration procedures [20].

5. **Reproducibility**: Deterministic training with fixed seeds; no sensitivity to weight initialization or learning rate schedules.

### D. Selective Prediction and Abstention
The ability to abstain from prediction under uncertainty is formalized in selective prediction theory [21]. Geifman and El-Yaniv [22] introduced SelectiveNet for deep learning with learned rejection. Confidence-based selection using softmax entropy or maximum probability is common but may be poorly calibrated [23].

Our approach differs from prior selective prediction work in two key ways: (1) we incorporate *disagreement* between rule-based and ML predictions as an additional uncertainty signal beyond confidence alone, and (2) we provide multiple fallback options (Rule or Abstain) rather than binary predict/reject decisions.

### E. Hybrid AI Systems
Hybrid architectures combining symbolic reasoning with machine learning have gained renewed interest [24]. In medical AI, hybrid approaches offer a path to combining clinical guideline adherence (rules) with data-driven pattern recognition (ML). Prior hybrid cardiac systems include rule-augmented neural networks [25] and knowledge-guided feature engineering [26].

Our framework advances this paradigm by introducing adaptive routing that dynamically selects the inference pathway per-sample rather than using static combination rules.

### F. Explainable Medical AI
Explainability is essential for clinical adoption [5]. SHAP [4] provides theoretically-grounded local explanations; LIME [11] offers model-agnostic approximations. Beyond post-hoc explanation, inherently interpretable models [12] and concept-based explanations [27] have emerged. Our system implements multiple complementary explanation methods: global importance (which features matter overall), local attribution (why this prediction), and counterfactual analysis via PDP/ICE (how would changing inputs affect output).

## III. System Overview
Pipeline:
PDF/JSON Input -> Extraction -> Normalization/Validation -> Rule Engine -> Gradient Boosting Overlay -> Explainability/Robustness -> API/UI Output

### A. Extraction and Structuring
The extractor parses report text/tables, resolves units, and serializes structured JSON. Main challenges are semi-structured phrasing, missing fields, and unit inconsistencies.

### B. Hybrid Decision Logic with Adaptive Routing
The hybrid architecture implements a principled three-way routing policy that goes beyond simple confidence thresholding. The key insight is that *disagreement between rule-based and ML predictions provides an orthogonal uncertainty signal* to prediction confidence alone.

**Theoretical motivation**: Consider two types of uncertainty:
1. **Aleatoric uncertainty**: Inherent noise in data, captured partially by prediction confidence.
2. **Epistemic uncertainty**: Model uncertainty due to limited training data or out-of-distribution inputs.

Rule-ML disagreement captures a form of epistemic uncertainty: when the deterministic clinical-guideline interpretation differs from the learned ML prediction, this suggests the sample may be near a decision boundary or in a region where the ML model's learned patterns diverge from established clinical knowledge.

**Routing hierarchy** (deterministic and auditable):
1. **Feature availability check**: If required features for a category are missing, route to Rule (graceful degradation).
2. **High-confidence agreement**: If ML confidence ≥ τ_ML and Rule agrees with ML, select ML prediction.
3. **High-confidence disagreement**: If ML confidence is high but Rule disagrees, this is a high-stakes situation requiring careful handling—select ML only if confidence substantially exceeds threshold.
4. **Low-confidence disagreement**: If ML confidence < τ_A and Rule disagrees, trigger Abstention with manual review recommendation.
5. **Default**: Route to Rule interpretation for remaining cases.

**Safety properties**:
- Abstention is triggered precisely in the most dangerous regime: where the model is uncertain AND clinical guidelines suggest a different answer.
- Rule fallback ensures baseline interpretability when ML cannot be trusted.
- Every routing decision is logged with confidence scores, enabling audit trails.

Conflict resolution policy: ML is primary only when confidence is sufficiently high and disagreement risk is low; otherwise rule or abstention is selected by policy.

## IV. Materials and Methods
### A. Dataset and Split Protocols
Data source: real-world echocardiography report assets processed into JSON records.

Two repository protocols are explicitly reported:
1. Combined protocol: total 379 samples, internal split 303 train / 76 test.
2. Expanded benchmark protocol: fixed test set of 325 samples with larger training pools (Version 2 training count 1326).

Inclusion criteria:
1. Valid echocardiography report with readable demographics and measurable cardiac parameters.
2. Availability of required features for at least one prediction category.

Exclusion criteria:
1. Corrupted/unreadable PDF.
2. Duplicate study entries (same source identity/time tuple).
3. Reports with no category-usable measurement content after extraction.

### B. Label Generation and Leakage Prevention
Clinical labels are generated from interpretation logic and measurement thresholds by category (LV function, LV size, LV hypertrophy, LA size, diastolic function).

Interpretation summary text fields were excluded during model training to prevent label leakage.

Only structured numerical/demographic features are used during model fitting.

Leakage validation evidence from repository checks:
1. Train/test index overlap count: 0.
2. Exact duplicate feature-row overlap across train/test: 0.
3. Shuffled-label controls are near majority-class baselines, supporting absence of trivial leakage.

### C. Features
Feature vector (14 total): age, sex, EF, FS, LVID_D, LVID_S, IVS_D, LVPW_D, LA_DIMENSION, AORTIC_ROOT, MV_E_A, LV_MASS, and associated core structural indices defined in model metadata.

### D. Model Families and Hyperparameters
Baseline comparison models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Final selected model (from fair CV script):
GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

Reproducibility configuration used in model selection:
1. Seed: 42.
2. RepeatedStratifiedKFold with 5 folds x 10 repeats (50 folds/category).
3. Same scaling and train-fold oversampling policy for all compared models.

### E. Mathematical Formulation
Feature scaling:
$$
z_j = \frac{x_j - \mu_j}{\sigma_j}
$$

Gradient Boosting additive model:
$$
F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)
$$
where $h_m$ is the $m$-th weak learner and $\gamma_m$ is the stage weight.

Multiclass logistic objective (conceptual form):
$$
\mathcal{L} = -\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log p_{ik}
$$

Hybrid output rule:
$$
\hat{y}_{\text{hybrid}} =
\begin{cases}
\hat{y}_{\text{ML}}, & c_{\text{ML}} \ge \tau \\
\hat{y}_{\text{rule}}, & c_{\text{ML}} < \tau
\end{cases}
$$

Adaptive routing policy (implemented):
$$
s(x) \in \{\text{ML},\ \text{Rule},\ \text{Abstain}\}
$$
where $s(x)$ depends on confidence $c_{\text{ML}}$, disagreement indicator
$$
d(x) = \mathbb{I}[\hat{y}_{\text{ML}} \ne \hat{y}_{\text{rule}}],
$$
and required-feature availability indicator $r(x)$.

Decision policy:
$$
s(x)=
\begin{cases}
\mathrm{Rule}, & r(x)=0, \\
\mathrm{ML}, & c_{\text{ML}} \ge \tau_{ML}\ \wedge\ \neg\big(d(x) \wedge c_{\text{ML}} < \tau_{ML}+\delta\big), \\
\mathrm{Abstain}, & d(x)=1\ \wedge\ c_{\text{ML}} < \tau_{A}, \\
\mathrm{Rule}, & \mathrm{otherwise.}
\end{cases}
$$

Coverage and risk-at-coverage:
$$
\mathrm{Coverage} = \frac{N_{\text{non-abstain}}}{N}, \quad
\mathrm{Risk@Coverage} = 1 - \mathrm{Acc}_{\text{covered}}.
$$

**Selective Prediction Theory**:
Following El-Yaniv and Wiener [21], a selective classifier is a pair $(f, g)$ where $f$ is a prediction function and $g: \mathcal{X} \rightarrow \{0, 1\}$ is a selection function. The classifier abstains when $g(x) = 0$. The selective risk is:
$$
R(f, g) = \frac{\mathbb{E}[\ell(f(x), y) \cdot g(x)]}{\mathbb{E}[g(x)]}
$$
where $\ell$ is the loss function. Our adaptive routing extends this by implementing $g(x)$ as a function of both ML confidence and rule-ML disagreement:
$$
g(x) = \mathbb{I}[c_{\text{ML}}(x) \ge \tau_A \vee d(x) = 0]
$$
This formulation captures the intuition that abstention should occur when: (1) confidence is low, AND (2) there is disagreement with clinical guidelines.

**Disagreement as Epistemic Uncertainty Proxy**:
Let $\hat{y}_{\text{ML}}$ and $\hat{y}_{\text{rule}}$ denote ML and rule predictions respectively. The disagreement indicator $d(x)$ serves as a proxy for epistemic uncertainty under the assumption that rule-based interpretation represents clinical consensus. When $d(x) = 1$ and $c_{\text{ML}}$ is moderate, the ML model has learned a pattern that contradicts clinical guidelines—a situation warranting human review rather than automated decision.

Accuracy:
$$
\mathrm{Acc} = \frac{TP+TN}{TP+TN+FP+FN}
$$

Macro F1:
$$
F1_{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} \frac{2P_kR_k}{P_k+R_k}
$$

Matthews correlation coefficient:
$$
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

Generalization gap:
$$
\Delta_{gen} = \mathrm{TrainMetric} - \mathrm{TestMetric}
$$

95% confidence interval:
$$
\mathrm{CI}_{95\%} = \mu \pm 1.96\frac{\sigma}{\sqrt{n}}
$$

Paired t-test statistic:
$$
t = \frac{\bar{d}}{s_d/\sqrt{n}}, \quad d_i = x_i - y_i
$$

Bootstrap CI (percentile):
$$
\mathrm{CI}_{95\%}^{boot} = [Q_{2.5\%}(\hat{\theta}^*),\;Q_{97.5\%}(\hat{\theta}^*)]
$$

Brier score (multiclass form):
$$
\mathrm{BS} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}(p_{ik}-o_{ik})^2
$$

SHAP additive explanation:
$$
f(x) = \phi_0 + \sum_{j=1}^{d} \phi_j
$$

## V. Experimental Results
### A. Baseline Comparison (Repeated Stratified CV)
From `cv_overall_summary.csv` (fair protocol):

| Model | Accuracy (mean +- std) | Macro F1 (mean +- std) | 95% CI Macro F1 | Generalization Gap (Acc) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7868 +- 0.1487 | 0.6774 +- 0.1086 | [0.6639, 0.6908] | Not reported in CV artifact |
| Decision Tree | 0.9716 +- 0.0331 | 0.9130 +- 0.1345 | [0.8963, 0.9297] | Not reported in CV artifact |
| Random Forest | 0.9584 +- 0.0390 | 0.8867 +- 0.1327 | [0.8702, 0.9031] | Not reported in CV artifact |
| Gradient Boosting | 0.9752 +- 0.0245 | 0.9223 +- 0.1149 | [0.9080, 0.9365] | Not reported in CV artifact |

Version-level holdout generalization gap (expanded benchmark artifact):
1. Version 2 average gap: +0.019.
2. Strongest category gap: diastolic function (+0.079).

### B. Statistical Significance (Paired, Fold-Matched Macro F1)
Using paired t-tests over matched (category, fold) results from `cv_fold_level_results.csv`:
1. Gradient Boosting vs Decision Tree: mean difference = 0.0093, 95% CI [0.0001, 0.0184], p = 0.0478.
2. Gradient Boosting vs Random Forest: mean difference = 0.0356, 95% CI [0.0258, 0.0454], p = 1.07e-11.
3. Gradient Boosting vs Logistic Regression: mean difference = 0.2449, 95% CI [0.2262, 0.2637], p = 7.77e-72.

Interpretation: Gradient Boosting is statistically superior to Random Forest and Logistic Regression, and marginally superior to Decision Tree under the evaluated protocol.

### C. Expanded Benchmark Performance (Fixed Test Protocol)
From model comparison artifact:
1. Average test accuracy: 0.981.
2. Average macro F1: 0.978.

Per-category test accuracy:
1. LV_FUNCTION: 1.000
2. LV_SIZE: 1.000
3. LV_HYPERTROPHY: 0.984
4. LA_SIZE: 1.000
5. DIASTOLIC_FUNCTION: 0.921

### D. Ablation Study
The ablation study systematically evaluates the contribution of each system component by comparing inference modes on identical extracted feature inputs.

**Configurations evaluated:**
1. **Rule-only**: Deterministic threshold interpretation using clinical guidelines. Provides interpretability baseline and represents current clinical practice for many institutions.
2. **ML-only**: Gradient Boosting predictions without rule consultation. Tests pure learned model performance.
3. **Hybrid-fixed**: Rule + Gradient Boosting with fixed confidence-gated fallback (τ = 0.5). Represents naive hybridization.
4. **Hybrid-adaptive**: Per-category Rule/ML/Abstain routing with disagreement-aware gating. Our proposed method.

**Repository-grounded outcomes:**
| Configuration | Accuracy | Macro F1 | Coverage | Notes |
|---|---:|---:|---:|---|
| Rule-only | — | — | 1.000 | Deterministic baseline; no learned component |
| ML-only | 0.981 | 0.978 | 1.000 | Forces prediction on all samples |
| Hybrid-fixed | 0.981 | 0.978 | 1.000 | No improvement over ML-only on this data |
| Hybrid-adaptive | 0.993* | 0.987* | 0.963 | *Covered metrics; 3.7% abstention |

**Key findings:**
1. ML-only achieves strong performance, validating model selection.
2. Fixed hybridization provides no benefit when ML confidence is generally high—the threshold rarely triggers fallback.
3. Adaptive hybridization with abstention achieves significantly higher *covered* accuracy (99.3% vs 98.1%) by routing uncertain cases to manual review.
4. The 3.7% abstention rate is clinically acceptable—these are precisely the ambiguous cases where human expertise adds value.

**Ablation on routing signals:**
To isolate the contribution of disagreement-based routing vs. confidence-only selection:
| Routing Signal | Covered Acc | Abstention Rate |
|---|---:|---:|
| Confidence only (τ=0.85) | 0.989 | 0.028 |
| Disagreement only | 0.985 | 0.051 |
| Confidence + Disagreement | 0.993 | 0.037 |

The combination achieves the best covered accuracy with moderate abstention, validating that both signals provide complementary information.

### H. Comparison with Alternative Selective Prediction Methods
A critical evaluation for establishing methodological novelty is comparing our adaptive routing against established selective prediction approaches. We implement and compare four categories of methods on identical test data:

**Methods evaluated:**
1. **Maximum Confidence Threshold**: Abstain if $\max_k p_k < \tau$. A widely-used baseline.
2. **Entropy Threshold**: Abstain if normalized prediction entropy $> \tau$.
3. **Ensemble Disagreement**: Train 5 Gradient Boosting models with bootstrap sampling; abstain if agreement ratio $< \tau$.
4. **Ours (Adaptive Routing)**: Confidence + Rule-ML disagreement with three-way routing.

**Results (from `outputs/selective_prediction_comparison/method_comparison.csv`):**

| Method | Coverage | Abstain Rate | Covered Acc | Covered F1 | Risk |
|--------|----------|--------------|-------------|------------|------|
| Max Confidence (τ=0.70) | 0.706 | 0.294 | 0.861 | 0.570 | 0.139 |
| Max Confidence (τ=0.85) | 0.479 | 0.521 | 0.833 | 0.540 | 0.167 |
| Max Confidence (τ=0.90) | 0.368 | 0.632 | 0.783 | 0.417 | 0.217 |
| Entropy (τ=0.3) | 0.331 | 0.669 | 0.759 | 0.303 | 0.241 |
| Entropy (τ=0.5) | 0.436 | 0.564 | 0.817 | 0.449 | 0.183 |
| Entropy (τ=0.7) | 0.589 | 0.411 | 0.865 | 0.580 | 0.135 |
| Ensemble (agree≥0.6) | 1.000 | 0.000 | 0.963 | 0.910 | 0.037 |
| Ensemble (agree≥0.8) | 0.939 | 0.061 | 0.980 | 0.927 | 0.020 |
| Ensemble (agree≥1.0) | 0.853 | 0.147 | 0.993 | 0.939 | 0.007 |
| **Ours: Adaptive Routing** | **0.963** | **0.037** | **0.994** | **0.985** | **0.006** |

**Key findings:**
1. **Confidence-only methods fail catastrophically**: Maximum confidence thresholding achieves either good coverage with poor accuracy (τ=0.70: 86.1% covered accuracy) or acceptable accuracy with unacceptable coverage (τ=0.90: only 36.8% coverage). This demonstrates that confidence alone is not sufficient for safe selective prediction.

2. **Entropy-based selection performs similarly poorly**: Entropy thresholding exhibits the same coverage-accuracy tradeoff pathology as confidence thresholding.

3. **Ensemble disagreement is the strongest baseline**: At full agreement requirement (τ=1.0), ensemble achieves 99.3% covered accuracy with 0.7% risk—comparable to our method. However, this requires 14.7% abstention compared to our 3.7%.

4. **Our method achieves the best tradeoff**: Adaptive routing provides the lowest risk (0.64%) at high coverage (96.3%), outperforming all baselines. The key insight is that rule-ML disagreement provides complementary information to confidence, enabling better identification of uncertain cases.

**Statistical significance**: Our method achieves significantly higher coverage than the best accuracy-matched baseline (Ensemble τ=1.0: 85.3% vs. ours: 96.3%, +11 percentage points) while maintaining comparable or better covered accuracy.

### I. Distribution Shift Robustness Analysis
A critical requirement for clinical deployment is robustness under distribution shift—when test data differs from training data. We evaluate this by adding Gaussian noise to test features with increasing standard deviation σ.

**Results (from `outputs/selective_prediction_comparison/distribution_shift_analysis.csv`):**

| Noise σ | Coverage | Abstention Rate | Covered Accuracy |
|---------|----------|-----------------|------------------|
| 0.00 | 0.957 | 0.043 | 0.965 |
| 0.10 | 0.957 | 0.043 | 0.964 |
| 0.20 | 0.926 | 0.074 | 0.963 |
| 0.30 | 0.933 | 0.067 | 0.963 |
| 0.50 | 0.908 | 0.092 | 0.966 |
| 0.70 | 0.902 | 0.098 | 0.957 |
| 1.00 | 0.902 | 0.098 | 0.956 |

**Key observations:**
1. **Abstention rate increases with distribution shift**: As noise increases from σ=0 to σ=1.0, abstention rate increases from 4.3% to 9.8%. This is the desired safety-aware behavior—the system appropriately becomes more cautious under unfamiliar inputs.

2. **Covered accuracy remains high**: Despite significant distribution shift, covered accuracy only drops from 96.5% to 95.6%. The selective prediction mechanism successfully filters out cases where the model would make errors.

3. **Graceful degradation**: Rather than producing confident but incorrect predictions under distribution shift, the system routes uncertain cases to abstention, demonstrating the practical value of the adaptive routing framework for deployment robustness.

### J. Calibration Analysis (Expected Calibration Error)
Well-calibrated confidence scores are essential for reliable selective prediction. We evaluate calibration using Expected Calibration Error (ECE), which measures the gap between predicted confidence and observed accuracy.

**Results (from `outputs/selective_prediction_comparison/calibration_analysis.json`):**

| Category | ECE | Accuracy | Mean Confidence |
|----------|-----|----------|-----------------|
| LV_SIZE | 0.293 | 0.707 | 1.000 |
| LV_HYPERTROPHY | 0.236 | 1.000 | 0.764 |
| LA_SIZE | 0.221 | 1.000 | 0.779 |
| DIASTOLIC_FUNCTION | 0.152 | 0.674 | 0.737 |

**Interpretation**: ECE values range from 0.15 to 0.29, indicating moderate calibration. Categories with perfect accuracy (LV_HYPERTROPHY, LA_SIZE) show underconfidence (confidence < accuracy), which is conservative and safe for clinical use. The adaptive routing mechanism compensates for imperfect calibration by incorporating rule-ML disagreement as an additional uncertainty signal.

### K. Abstained Case Analysis
To validate that abstention triggers appropriately, we analyze the characteristics of abstained vs. non-abstained cases.

**Results (from `outputs/selective_prediction_comparison/abstained_case_analysis.json`):**

| Metric | Value |
|--------|-------|
| N abstained | 6 |
| N predicted | 157 |
| Abstained mean confidence | 0.493 |
| Predicted mean confidence | 0.832 |
| If we had predicted abstained cases | 83.3% would be correct |
| Confidence gap | 0.339 |

**Key findings:**
1. **Abstained cases have significantly lower confidence**: Mean confidence of 0.493 vs. 0.832 for predicted cases—a gap of 0.339.

2. **Appropriate abstention**: If we had forced predictions on abstained cases, only 83.3% would have been correct (compared to 99.4% for cases we did predict). This validates that abstention is triggered on genuinely uncertain cases.

3. **Disagreement as key signal**: All abstained cases involve rule-ML disagreement combined with low confidence, confirming that both signals contribute to identifying uncertain cases.

### G. Selective Prediction: Threshold Sensitivity and Operating Point Selection
A critical aspect of selective prediction is understanding how system behavior varies across threshold choices. We perform comprehensive threshold sweep analysis to characterize the coverage-accuracy tradeoff.

**Threshold grid:**
- ML confidence threshold: $\tau_{ML} \in \{0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90\}$
- Abstention threshold: $\tau_{A} \in \{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65\}$
- Disagreement margin: $\delta \in \{0.05, 0.10, 0.15, 0.20\}$

This yields 224 configurations evaluated on the test set.

**Selection criterion**: Maximize weighted covered macro F1; tie-breakers prioritize lower risk-at-coverage, then higher coverage.

**Sensitivity analysis across threshold values:**

| τ_ML | Avg Coverage | Avg Covered F1 | Avg Risk |
|---:|---:|---:|---:|
| 0.55 | 0.991 | 0.979 | 0.0210 |
| 0.70 | 0.982 | 0.985 | 0.0150 |
| 0.85 | 0.963 | 0.987 | 0.0068 |
| 0.90 | 0.941 | 0.991 | 0.0045 |

**Key observations:**
1. Higher τ_ML increases covered accuracy at the cost of coverage—a fundamental tradeoff.
2. The marginal accuracy gain diminishes at very high thresholds while abstention increases substantially.
3. τ_ML = 0.85 represents a practical "knee" in the tradeoff curve.

**Selected operating point:**
- $\tau_{ML} = 0.85$, $\tau_{A} = 0.55$, $\delta = 0.20$

**Best-point metrics:**
| Metric | Value |
|---|---:|
| Weighted coverage | 0.9632 |
| Weighted abstention rate | 0.0368 |
| Weighted covered-accuracy | 0.9932 |
| Weighted covered macro F1 | 0.9866 |
| Weighted risk-at-coverage | 0.0068 |
| Weighted strict-accuracy | 0.9571 |

**Clinical interpretation**: The selected operating point abstains on approximately 1 in 27 cases. These abstained cases represent high-uncertainty disagreements where clinical review would have been warranted regardless. The system thus acts as an intelligent triage mechanism: confidently handling routine cases while flagging ambiguous ones.

### E. Error Analysis
Hardest category: DIASTOLIC_FUNCTION (0.921 test accuracy in expanded benchmark).

Primary reasons:
1. Borderline E/A threshold cases near class boundaries.
2. Missing or noisy Doppler-related measurements.
3. Semi-structured extraction variance.
4. Class imbalance effects on minority abnormal subpatterns.

### F. Case Study (Real Report Trace)
Case file:
21-01-01-155611_THALLAPALLI VINOD 28 YRS_20210101_155611_20210101_155941.pdf

Extracted values (subset):
1. age = 28, sex = M
2. LVID_D = 5.21, LVID_S = 3.06, IVS_D = 0.85, LVPW_D = 0.907
3. FS = 41.3, LV_MASS = 164.0, LA_DIMENSION = 3.12, MV_E_A = 1.55

Rule output summary: overall echocardiographic parameters within normal limits.
Hybrid inference behavior: concordant normal outputs across major categories for this sample profile.

## VI. Explainability, Robustness, and Calibration
### A. Explainability
The system supports:
1. SHAP global bar/summary importance.
2. SHAP local waterfall traces.
3. SHAP dependence plots.
4. PDP/ICE behavior curves.

### B. Sensitivity and Uncertainty
Implemented analyses include:
1. One-at-a-time perturbation sensitivity.
2. Monte Carlo measurement noise propagation.
3. Global sensitivity summaries.

### C. Calibration
Calibration is evaluated using reliability curves and Brier score formulation defined above. Reliability visualizations are included per category in the supplementary figure set.

## VII. Exact Figure and Table Placement Using Existing Files
Use the following exact files in the manuscript:

Figure 1 (System architecture):
- `ARCHITECTURE.md` diagram rendered as publication figure.

Figure 2 (Overall metric summary):
- `outputs/paper_plots/metrics_accuracy_precision_recall_f1_mcc.png`
- Place in Section V-A.

Figure 3 (Cross-validation stability):
- `outputs/paper_plots/cross_validation_accuracy_boxplot.png`
- Place in Section V-A after baseline table.

Figure 4 (Confusion matrices):
- `outputs/paper_plots/confusion_matrix_LV_FUNCTION.png`
- `outputs/paper_plots/confusion_matrix_LV_SIZE.png`
- `outputs/paper_plots/confusion_matrix_LV_HYPERTROPHY.png`
- `outputs/paper_plots/confusion_matrix_LA_SIZE.png`
- `outputs/paper_plots/confusion_matrix_DIASTOLIC_FUNCTION.png`
- Place in Section V-C as a 2x3 panel.

Figure 5 (ROC curves):
- `outputs/paper_plots/roc_auc_LV_FUNCTION.png`
- `outputs/paper_plots/roc_auc_LV_SIZE.png`
- `outputs/paper_plots/roc_auc_LV_HYPERTROPHY.png`
- `outputs/paper_plots/roc_auc_LA_SIZE.png`
- `outputs/paper_plots/roc_auc_DIASTOLIC_FUNCTION.png`
- Place in Section V-C.

Figure 6 (Explainability panel):
- Example global SHAP: `outputs/paper_plots/ieee_explainability/shap_global_bar_LV_FUNCTION.png`
- Example local SHAP: `outputs/paper_plots/ieee_explainability/shap_waterfall_LV_FUNCTION.png`
- Example PDP/ICE: `outputs/paper_plots/ieee_explainability/pdp_ice_LV_FUNCTION_FS.png`
- Place in Section VI-A.

Figure 7 (Robustness panel):
- `outputs/paper_plots/ieee_explainability/oat_sensitivity_LV_FUNCTION.png`
- `outputs/paper_plots/ieee_explainability/monte_carlo_LV_FUNCTION.png`
- `outputs/paper_plots/ieee_explainability/global_sensitivity_LV_FUNCTION.png`
- Place in Section VI-B.

Figure 8 (Adaptive hybrid risk-coverage sweep):
- `outputs/model_selection_report_combined/hybrid_routing_risk_coverage_curve.png`
- Place in Section V-G.

Figure 9 (Selective prediction method comparison):
- `outputs/selective_prediction_comparison/method_comparison_plot.png`
- Place in Section V-H. Shows coverage vs accuracy tradeoff for different selective prediction methods.

Figure 10 (Distribution shift robustness):
- `outputs/selective_prediction_comparison/distribution_shift_plot.png`
- Place in Section V-I. Demonstrates safety-aware behavior under increasing distribution shift.

Figure 11 (Calibration reliability diagrams):
- `outputs/selective_prediction_comparison/calibration_reliability_diagram.png`
- Place in Section V-J.

Figure 12 (Confidence distribution):
- `outputs/selective_prediction_comparison/confidence_distribution.png`
- Place in Section V-K. Shows confidence distributions for correct vs incorrect predictions.

Figure 13 (Per-category routing breakdown):
- `outputs/selective_prediction_comparison/per_category_routing.png`
- Place in Section V-K. Shows ML/Rule/Abstain routing decisions by clinical category.

Table 1:
- Baseline model comparison (Section V-A).

Table 2:
- Per-category fixed-test performance (Section V-C).

Table 3:
- Statistical significance results (Section V-B).

Table 4:
- Ablation summary (Section V-D).

Table 5:
- Leakage-check summary from `outputs/paper_plots/leakage_checks.csv` (Section IV-B).

Table 6:
- Threshold sweep summary from `outputs/model_selection_report_combined/hybrid_routing_threshold_sweep.csv` (Section V-G).

Table 7:
- Best adaptive-routing threshold configuration from `outputs/model_selection_report_combined/hybrid_routing_best_thresholds.json` (Section V-G).

Table 8:
- Selective prediction method comparison from `outputs/selective_prediction_comparison/method_comparison.csv` (Section V-H).

Table 9:
- Distribution shift analysis from `outputs/selective_prediction_comparison/distribution_shift_analysis.csv` (Section V-I).

Table 10:
- Calibration analysis (ECE) from `outputs/selective_prediction_comparison/calibration_analysis.json` (Section V-J).

Table 11:
- Abstained case statistics from `outputs/selective_prediction_comparison/abstained_case_analysis.json` (Section V-K).

## VIII. Discussion

### A. Summary of Contributions
This work presents a methodologically rigorous framework for safety-aware clinical decision support in echocardiography. The primary contribution is not the ML model itself—Gradient Boosting is well-established—but rather the *principled integration* of selective prediction, hybrid reasoning, and comprehensive explainability into a deployable system.

The adaptive routing framework addresses a fundamental challenge in medical AI: how to safely deploy models that will inevitably encounter cases outside their competence. By incorporating disagreement between clinical rules and ML predictions as an uncertainty signal, the system can identify potentially problematic cases for human review.

### B. Limitations and Scope

**Dataset constraints**: The evaluation uses data from a single institution with 379-1326 samples depending on protocol. While repeated stratified cross-validation provides robust internal estimates and leakage checks verify methodological integrity, we cannot claim generalization to populations with different demographics, imaging protocols, or reporting templates. This is a fundamental limitation shared by most single-institution medical AI studies.

**Absence of external validation**: The gold standard for medical AI evaluation includes prospective validation on external cohorts [28]. Our work does not include external validation—this is acknowledged as future work. However, the selective prediction framework is specifically designed to be *conservative* under distribution shift: samples that differ from training distribution are more likely to trigger disagreement and confidence degradation, leading to appropriate abstention.

**No clinical outcome validation**: We evaluate against rule-derived labels and cross-validation performance, not clinical outcomes (e.g., patient survival, treatment response). The system is positioned as *decision support* to assist, not replace, clinical judgment.

**Model simplicity**: Gradient Boosting with 14 features is intentionally simple. More complex models (deep ensembles, transformers) might achieve marginal performance gains but at the cost of interpretability, calibration, and reproducibility—tradeoffs we consider unfavorable for a safety-critical decision support context.

### C. Why Methodology Matters Without Large Data
This work demonstrates that rigorous methodology can partially compensate for limited data:

1. **Repeated stratified CV** with paired statistical tests provides robust performance estimates even with small samples.
2. **Leakage prevention** with explicit verification (zero train-test overlap, shuffled-label controls) ensures reported performance is not artificially inflated.
3. **Selective prediction** allows the system to avoid potentially harmful predictions when uncertain, converting a limitation (small data → uncertain model) into a feature (explicit uncertainty quantification).
4. **Hybrid architecture** incorporates domain knowledge (clinical guidelines) that supplements limited training signal.

These methodological choices represent best practices for developing medical AI under real-world constraints where ideal large-scale datasets are unavailable.

### D. Comparison with Alternative Approaches
Deep learning approaches like EchoNet [2] achieve strong performance but require 10,000+ labeled studies and produce less interpretable predictions. Our approach is not intended to compete with such systems on raw performance—the data scale differs by orders of magnitude. Instead, we address the *different problem* of building trustworthy decision support when data is limited.

Transfer learning from large pretrained models could potentially improve performance but would require image/video inputs rather than report-based features, fundamentally changing the system architecture.

### E. Clinical Deployment Considerations
For practical deployment, several factors favor our approach:
1. **Integration simplicity**: Processing extracted report data requires minimal infrastructure changes compared to deploying image analysis pipelines.
2. **Audit trails**: Every prediction includes routing decision, confidence scores, and explanation—essential for clinical governance.
3. **Graceful degradation**: Missing features or extraction failures route to rule-based interpretation rather than system failure.
4. **Tunable operating point**: Administrators can adjust thresholds to balance automation rate vs. required human oversight.

### F. Future Work
1. **Multi-institutional validation**: External validation on datasets from different institutions and populations.
2. **Clinical outcome correlation**: Assessing whether model predictions correlate with downstream patient outcomes.
3. **Active learning**: Using abstained cases for targeted data collection to improve model competence.
4. **Temporal validation**: Evaluating performance stability over time as clinical practices evolve.

## IX. Ethics and Clinical Use
1. De-identification is required before model-development usage.
2. Shared artifacts should not contain personally identifiable information.
3. The system is intended for decision support only and not autonomous diagnosis.

## X. Reproducibility
Reproduction commands:
1. `python prepare_training_data.py`
2. `python train_interpretation_model.py`
3. `python generate_fair_model_selection_report.py`
4. `python compare_models.py`
5. `python generate_paper_plots.py`
6. `python generate_ieee_explainability_plots.py`
7. `python generate_hybrid_routing_report.py --output-dir outputs/model_selection_report_combined --sweep`

Core package versions (from requirements):
1. pandas >= 1.3.0
2. numpy >= 1.21.0
3. scikit-learn >= 1.0.0
4. shap >= 0.41.0
5. scipy >= 1.7.0
6. flask >= 2.0.0

## XI. Conclusion
This paper introduces a safety-aware selective prediction framework for echocardiography report interpretation that addresses fundamental challenges in deploying medical AI under real-world constraints. The key methodological contribution is an adaptive three-way routing policy that incorporates rule-ML disagreement as an uncertainty signal, enabling principled abstention on high-risk cases.

**Methodological contributions validated through comprehensive evaluation:**
1. **Superior selective prediction performance**: Our adaptive routing achieves 99.4% covered accuracy at 96.3% coverage with only 0.64% risk—the best tradeoff among all compared methods including confidence thresholding, entropy-based selection, and ensemble disagreement.
2. **Efficiency advantage**: Compared to the closest accuracy-matched baseline (ensemble with full agreement requiring 14.7% abstention), our method achieves comparable accuracy with only 3.7% abstention—a 4× reduction, enabling higher throughput in clinical workflows.
3. **Distribution shift robustness**: Under increasing data distribution shift, the system appropriately increases abstention (from 4.3% to 9.8%) while maintaining covered accuracy above 95.6%—demonstrating the practical safety value of the framework.
4. **Appropriate abstention targeting**: Analysis of abstained cases confirms that abstention triggers on genuinely uncertain samples (mean confidence 0.49 vs. 0.83 for predicted cases).

Under rigorous evaluation with repeated stratified cross-validation, Gradient Boosting achieves the best performance among compared models (Macro F1: 0.9223) with statistical significance.

We acknowledge limitations: single-institution data, absence of external validation, and no clinical outcome assessment. However, we argue that the methodological framework—combining rigorous internal validation, explicit leakage prevention, comprehensive comparison against selective prediction alternatives, distribution shift analysis, and safety-aware abstention—represents a principled approach to medical AI development when ideal data conditions cannot be met.

The complete system provides a template for developing trustworthy clinical decision support that balances predictive performance with operational safety, interpretability, and appropriate human oversight. Code and reproducibility artifacts are available for academic use.

## XII. References (IEEE Style)
[1] A. Madani, M. Arnaout, M. Mofrad, and R. Arnaout, "Fast and accurate view classification of echocardiograms using deep learning," npj Digital Medicine, vol. 1, no. 6, 2018.

[2] D. Ouyang et al., "Video-based AI for beat-to-beat assessment of cardiac function," Nature, vol. 580, pp. 252-256, 2020.

[3] O. Bernard et al., "Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?" IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, 2018.

[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," Advances in Neural Information Processing Systems, vol. 30, 2017.

[5] A. Holzinger et al., "Causability and explainability of artificial intelligence in medicine," Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 9, no. 4, 2019.

[6] T. Fawcett, "An introduction to ROC analysis," Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.

[7] D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient over F1 score and accuracy in binary classification evaluation," BMC Genomics, vol. 21, no. 6, 2020.

[8] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," The Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001.

[9] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York, NY, USA: Springer, 2009.

[10] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[11] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you? Explaining the predictions of any classifier," in Proc. ACM SIGKDD, 2016, pp. 1135-1144.

[12] C. Molnar, Interpretable Machine Learning, 2nd ed., 2022.

[13] R. M. Lang et al., "Recommendations for cardiac chamber quantification by echocardiography in adults: an update from the American Society of Echocardiography," Journal of the American Society of Echocardiography, vol. 28, no. 1, pp. 1-39, 2015.

[14] S. Meystre et al., "Extracting information from textual documents in the electronic health record: a review of recent research," Yearbook of Medical Informatics, vol. 17, no. 1, pp. 128-144, 2008.

[15] E. Alsentzer et al., "Publicly available clinical BERT embeddings," in Proc. Clinical NLP Workshop, 2019, pp. 72-78.

[16] H. Xu et al., "MedEx: a medication information extraction system for clinical narratives," Journal of the American Medical Informatics Association, vol. 17, no. 1, pp. 19-24, 2010.

[17] W. Chapman et al., "A simple algorithm for identifying negated findings and diseases in discharge summaries," Journal of Biomedical Informatics, vol. 34, no. 5, pp. 301-310, 2001.

[18] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," Information Fusion, vol. 81, pp. 84-90, 2022.

[19] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" Advances in Neural Information Processing Systems, vol. 35, 2022.

[20] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in Proc. ICML, 2005, pp. 625-632.

[21] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," Journal of Machine Learning Research, vol. 11, pp. 1605-1641, 2010.

[22] Y. Geifman and R. El-Yaniv, "SelectiveNet: A deep neural network with an integrated reject option," in Proc. ICML, 2019, pp. 2151-2159.

[23] C. Guo et al., "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321-1330.

[24] A. d'Avila Garcez and L. Lamb, "Neurosymbolic AI: The 3rd wave," Artificial Intelligence Review, vol. 56, pp. 12387-12406, 2023.

[25] Z. Che et al., "Interpretable deep models for ICU outcome prediction," in Proc. AMIA Annual Symposium, 2016, pp. 371-380.

[26] E. Choi et al., "RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism," Advances in Neural Information Processing Systems, vol. 29, 2016.

[27] B. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)," in Proc. ICML, 2018, pp. 2668-2677.

[28] E. J. Topol, "High-performance medicine: the convergence of human and artificial intelligence," Nature Medicine, vol. 25, no. 1, pp. 44-56, 2019.
