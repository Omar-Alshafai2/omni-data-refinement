"""
Rule-based AI Assistant Engine.

Analyzes the diagnostic context (issues, profile, bias, leakage) and
produces natural-language answers to common data science questions
without requiring any external AI API.
"""
import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DiagnosticReport, ProfilingReport, BiasReport, LeakageReport


RESPONSE_SEPARATOR = "─" * 60


class AIAssistantEngine:
    """
    A rule-based virtual data consultant that answers questions
    about your dataset in plain English.
    """

    def __init__(self):
        self._handlers = [
            ("model perform", self._why_poor_performance),
            ("perform",       self._why_poor_performance),
            ("missing",       self._about_missing),
            ("duplicate",     self._about_duplicates),
            ("outlier",       self._about_outliers),
            ("imbalance",     self._about_imbalance),
            ("bias",          self._about_bias),
            ("leak",          self._about_leakage),
            ("drift",         self._about_drift),
            ("clean",         self._about_cleaning),
            ("quality",       self._about_quality),
            ("ready",         self._about_ml_readiness),
            ("feature",       self._about_features),
            ("type",          self._about_types),
        ]

    def ask(self,
            question: str,
            diagnostic_report=None,
            profile=None,
            bias_report=None,
            leakage_report=None) -> str:
        """
        Answers a plain-English question about the dataset.

        Args:
            question: Any natural language question about the data.
            diagnostic_report: The DiagnosticReport if diagnose() was run.
            profile: The ProfilingReport if profile() was run.
            bias_report: The BiasReport if detect_bias() was run.
            leakage_report: The LeakageReport if detect_leakage() was run.

        Returns:
            A detailed, human-readable answer as a string.
        """
        q_lower = question.lower()

        for keyword, handler in self._handlers:
            if keyword in q_lower:
                return handler(diagnostic_report, profile, bias_report, leakage_report)

        return self._general_advice(diagnostic_report, profile, bias_report, leakage_report)

    # ── Handlers ────────────────────────────────────────────────────────────

    def _why_poor_performance(self, diag, profile, bias, leakage) -> str:
        causes = []
        if diag and diag.issues:
            issue_types = {i.issue_type for i in diag.issues}
            if "Missing Values" in issue_types:
                causes.append("1. Missing Values\n   Your dataset has missing values, which can confuse models.")
            if "Duplicate Rows" in issue_types:
                causes.append("2. Duplicate Rows\n   Duplicates can cause models to over-fit to repeated examples.")
            if "Mixed Data Types" in issue_types:
                causes.append("3. Mixed Data Types\n   Mixed types prevent models from interpreting columns correctly.")

        if bias and bias.issues:
            for i, b in enumerate(bias.issues[:2], len(causes) + 1):
                causes.append(f"{i}. {b.issue_type} in '{b.column}'\n   {b.description}")

        if leakage and leakage.issues:
            causes.append(f"{len(causes)+1}. Data Leakage\n   '{leakage.issues[0].feature}' may be leaking target information.")

        if not causes:
            return (
                "Based on available diagnostics, no obvious data issues were found.\n"
                "Consider:\n"
                "  • Running diagnose() to check for data quality issues.\n"
                "  • Reviewing your model hyperparameters and architecture.\n"
                "  • Increasing your training data size.\n"
                "  • Trying feature engineering or feature selection."
            )

        return "Possible causes for poor model performance:\n\n" + "\n\n".join(causes) + (
            "\n\nRecommendation: Run assistant.fix() to resolve data quality issues first."
        )

    def _about_missing(self, diag, profile, bias, leakage) -> str:
        if not diag:
            return "Run assistant.diagnose() first to detect missing values."
        mv = [i for i in diag.issues if i.issue_type == "Missing Values"]
        if not mv:
            return "✓ No missing values were detected in your dataset."
        lines = [f"Found {len(mv)} column(s) with missing values:\n"]
        for i in mv:
            lines.append(f"  • {i.column}: {i.description}")
        lines.append("\nOMR will impute numeric columns with the median and categorical columns with 'Unknown'.")
        lines.append("Run assistant.fix() to apply these fixes automatically.")
        return "\n".join(lines)

    def _about_duplicates(self, diag, profile, bias, leakage) -> str:
        if not diag:
            return "Run assistant.diagnose() first to detect duplicates."
        dups = [i for i in diag.issues if i.issue_type == "Duplicate Rows"]
        if not dups:
            return "✓ No duplicate rows were detected."
        i = dups[0]
        return (
            f"Found duplicate rows: {i.description}\n\n"
            f"Duplicates can cause:\n"
            f"  • Models to overfit to repeated patterns.\n"
            f"  • Inflated evaluation metrics.\n"
            f"  • Incorrect statistical summaries.\n\n"
            f"Run assistant.fix() to remove them automatically."
        )

    def _about_outliers(self, diag, profile, bias, leakage) -> str:
        return (
            "Outlier detection is available via:\n\n"
            "  results = assistant.detect_outliers(method='iqr')   # or 'zscore'\n"
            "  assistant.remove_outliers(method='iqr', strategy='cap')\n\n"
            "Strategies:\n"
            "  • flag   — identify outliers without changing the data\n"
            "  • remove — drop rows containing outliers\n"
            "  • cap    — winsorize (clip) values to IQR fences\n\n"
            "IQR is recommended for skewed data; Z-Score works best for normally distributed data."
        )

    def _about_imbalance(self, diag, profile, bias, leakage) -> str:
        if not bias or not bias.issues:
            return (
                "No class imbalance detected. Run assistant.detect_bias(target_col='your_target') "
                "to check for imbalance in your target column."
            )
        imbalance = [i for i in bias.issues if i.issue_type == "Class Imbalance"]
        if not imbalance:
            return "✓ No class imbalance detected in the target column."
        b = imbalance[0]
        return (
            f"Class imbalance detected in '{b.column}':\n"
            f"  {b.description}\n\n"
            f"Recommended approaches:\n"
            f"  • Use class_weight='balanced' in sklearn models.\n"
            f"  • Oversample the minority class (SMOTE from imbalanced-learn).\n"
            f"  • Undersample the majority class.\n"
            f"  • Use precision/recall/F1 instead of accuracy for evaluation."
        )

    def _about_bias(self, diag, profile, bias, leakage) -> str:
        return self._about_imbalance(diag, profile, bias, leakage)

    def _about_leakage(self, diag, profile, bias, leakage) -> str:
        if not leakage or not leakage.issues:
            return (
                "No leakage detected. Run assistant.detect_leakage(target_col='your_target') "
                "to check for data leakage in your features."
            )
        l = leakage.issues[0]
        return (
            f"Data leakage detected:\n"
            f"  Feature: '{l.feature}'\n"
            f"  Correlation with target: {l.correlation:.4f} ({l.risk} risk)\n\n"
            f"Why is this dangerous?\n"
            f"  If '{l.feature}' directly encodes or derives from '{l.target}', "
            f"your model will learn a shortcut that won't generalize to new data. "
            f"This causes artificially high training performance and poor production performance.\n\n"
            f"Action: Drop or review the feature '{l.feature}' before training."
        )

    def _about_drift(self, diag, profile, bias, leakage) -> str:
        return (
            "Data drift compares your training data to production data.\n\n"
            "Usage:\n"
            "  report = assistant.detect_drift(production_df)\n\n"
            "OMR will compare the mean and distribution of each numeric column "
            "between the two datasets and flag columns where significant changes occurred.\n\n"
            "High drift means your model may underperform in production because "
            "the data it sees at inference time looks different from what it was trained on."
        )

    def _about_cleaning(self, diag, profile, bias, leakage) -> str:
        return (
            "OMR automated cleaning pipeline:\n\n"
            "  1. assistant.diagnose()  — identify issues\n"
            "  2. assistant.fix()       — auto-resolve all detected issues\n\n"
            "What fix() does:\n"
            "  • Replaces string-based missing indicators (N/A, null, ?) with NaN\n"
            "  • Drops exact duplicate rows\n"
            "  • Imputes numeric missing values with the column median\n"
            "  • Imputes categorical missing values with 'Unknown'\n"
            "  • Converts mixed-type columns to strings\n"
            "  • Auto-converts string-encoded numbers and dates\n\n"
            "Run assistant.explain() after fix() to see a full change log."
        )

    def _about_quality(self, diag, profile, bias, leakage) -> str:
        if not diag:
            return "Run assistant.diagnose() to get a data quality assessment."
        total_issues = len(diag.issues)
        high = sum(1 for i in diag.issues if i.severity == "High")
        medium = sum(1 for i in diag.issues if i.severity == "Medium")
        low = sum(1 for i in diag.issues if i.severity == "Low")
        score = max(0, 100 - (high * 20) - (medium * 10) - (low * 3))
        return (
            f"Dataset Quality Assessment:\n\n"
            f"  Estimated Score: {score}/100\n"
            f"  Total Issues: {total_issues}\n"
            f"    High severity:   {high}\n"
            f"    Medium severity: {medium}\n"
            f"    Low severity:    {low}\n\n"
            f"  Run assistant.fix() to resolve issues and improve this score."
        )

    def _about_ml_readiness(self, diag, profile, bias, leakage) -> str:
        blockers = []
        warnings = []

        if diag:
            for i in diag.issues:
                if i.severity == "High":
                    blockers.append(f"• {i.issue_type} in '{i.column}'")
                else:
                    warnings.append(f"• {i.issue_type} in '{i.column}'")

        if leakage and leakage.issues:
            blockers.append(f"• Data Leakage risk in '{leakage.issues[0].feature}'")

        if bias and any(i.issue_type == "Class Imbalance" for i in bias.issues):
            warnings.append("• Class imbalance detected in target column")

        if blockers:
            status = "❌ NOT READY"
        elif warnings:
            status = "⚠️ READY WITH WARNINGS"
        else:
            status = "✅ ML READY"

        lines = [f"ML Readiness: {status}\n"]
        if blockers:
            lines.append("Blockers (must fix):\n" + "\n".join(blockers))
        if warnings:
            lines.append("Warnings (recommended to fix):\n" + "\n".join(warnings))
        if not blockers and not warnings:
            lines.append("Your dataset appears ready for machine learning!")

        return "\n\n".join(lines)

    def _about_features(self, diag, profile, bias, leakage) -> str:
        return (
            "Feature engineering suggestions from OMR:\n\n"
            "  • Numeric → Log transform for right-skewed columns\n"
            "  • Age column → Age groups (0-18, 18-35, 35-60, 60+)\n"
            "  • Date column → Year, Month, Day, DayOfWeek, IsWeekend\n"
            "  • Timestamp → Hour, Minute, IsBusinessHour\n"
            "  • High cardinality strings → Target encoding or frequency encoding\n"
            "  • Correlated columns → Principal Component Analysis (PCA)\n\n"
            "Automatic feature engineering (suggest_features) is coming in a future OMR release."
        )

    def _about_types(self, diag, profile, bias, leakage) -> str:
        if not diag:
            return "Run assistant.diagnose() to detect mixed type issues."
        mixed = [i for i in diag.issues if i.issue_type == "Mixed Data Types"]
        if not mixed:
            return "✓ No mixed data type issues detected."
        lines = [f"Found {len(mixed)} column(s) with mixed data types:\n"]
        for i in mixed:
            lines.append(f"  • {i.column}: {i.description}")
        lines.append("\nRun assistant.fix() to cast these columns to a uniform string type.")
        return "\n".join(lines)

    def _general_advice(self, diag, profile, bias, leakage) -> str:
        return (
            "I can answer questions about:\n\n"
            "  • 'Why is my model performing poorly?'\n"
            "  • 'What missing values do I have?'\n"
            "  • 'Are there duplicates?'\n"
            "  • 'How do I detect outliers?'\n"
            "  • 'Is there class imbalance?'\n"
            "  • 'Is there data leakage?'\n"
            "  • 'What is the data quality score?'\n"
            "  • 'Is my dataset ML-ready?'\n"
            "  • 'What feature engineering should I do?'\n\n"
            "Try: assistant.ask('Why is my model performing poorly?')"
        )
