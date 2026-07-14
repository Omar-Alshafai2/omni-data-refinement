"""
OMR Explainability Module — Rule-based explanations for data quality issues.
"""

EXPLANATIONS = {
    "class_imbalance": {
        "definition": "One class has significantly fewer samples than the other(s).",
        "why_it_matters": "Models will become biased toward the majority class, predicting it almost always to minimize error, while completely failing on the minority class.",
        "risks": "High accuracy but terrible recall/precision on the class you actually care about (e.g., fraud detection).",
        "recommended_fixes": "1. Use SMOTE (Synthetic Minority Over-sampling Technique)\n2. Undersample the majority class\n3. Use class weights in your model (e.g., class_weight='balanced')",
    },
    "data_leakage": {
        "definition": "A feature in the training data shares highly correlated or direct information with the target variable, which won't be available in production.",
        "why_it_matters": "The model 'cheats' by learning the answer directly from the feature. It will look like it has 99% accuracy during training, but will fail completely in the real world.",
        "risks": "Overly optimistic model evaluation leading to disastrous production deployments.",
        "recommended_fixes": "1. Remove the leaking feature.\n2. Ensure features are only derived from data available *before* the target outcome occurs.",
    },
    "outliers": {
        "definition": "Data points that differ significantly from other observations.",
        "why_it_matters": "Algorithms like Linear Regression and K-Means are highly sensitive to outliers, which can skew the model's coefficients heavily.",
        "risks": "Poor generalization, skewed predictions, and high mean squared error.",
        "recommended_fixes": "1. Winsorize (cap) the extreme values.\n2. Remove the rows if they are data entry errors.\n3. Use robust algorithms (e.g., Random Forest, XGBoost) that handle outliers naturally.",
    },
    "multicollinearity": {
        "definition": "Two or more predictor variables are highly correlated with each other.",
        "why_it_matters": "It makes it difficult to determine the individual effect of each feature on the target variable. The model coefficients become unstable and highly sensitive to small changes in the model.",
        "risks": "Unreliable feature importance, inflated variance of coefficient estimates.",
        "recommended_fixes": "1. Remove one of the highly correlated features.\n2. Use Principal Component Analysis (PCA) to combine them.\n3. Use regularization techniques like Ridge or Lasso regression.",
    }
}


class ExplainabilityEngine:
    """Provides rule-based explanations for data quality issues."""

    def run(self, issue: str) -> dict:
        """Returns the explanation dictionary for a given issue."""
        key = issue.lower().replace(" ", "_")
        if key in EXPLANATIONS:
            return EXPLANATIONS[key]
        return {
            "definition": "Unknown issue type.",
            "why_it_matters": "N/A",
            "risks": "N/A",
            "recommended_fixes": "N/A",
        }
