"""
HealthSense Association Rule Mining Module.
Implements the Apriori algorithm via MLxtend to uncover transactional health patterns.
Discretizes physiological continuous variables and calculates Support, Confidence, and Lift.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from utils.config import THEME


def prepare_transactional_data(df: pd.DataFrame, target_column: str = "Outcome") -> pd.DataFrame:
    """
    Discretize continuous health variables into one-hot encoded binary transactions.
    """
    trans_df = pd.DataFrame(index=df.index)

    # 1. Glucose Discretization
    if "Glucose" in df.columns:
        trans_df["Glucose_Normal"] = (df["Glucose"] < 140).astype(int)
        trans_df["Glucose_Impaired"] = ((df["Glucose"] >= 140) & (df["Glucose"] < 200)).astype(int)
        trans_df["Glucose_High"] = (df["Glucose"] >= 200).astype(int)

    # 2. BMI Discretization
    if "BMI" in df.columns:
        trans_df["BMI_Normal"] = (df["BMI"] < 25.0).astype(int)
        trans_df["BMI_Overweight"] = ((df["BMI"] >= 25.0) & (df["BMI"] < 30.0)).astype(int)
        trans_df["BMI_Obese"] = (df["BMI"] >= 30.0).astype(int)

    # 3. Age Discretization
    if "Age" in df.columns:
        trans_df["Age_Young(<30)"] = (df["Age"] < 30).astype(int)
        trans_df["Age_Middle(30-50)"] = ((df["Age"] >= 30) & (df["Age"] <= 50)).astype(int)
        trans_df["Age_Senior(>50)"] = (df["Age"] > 50).astype(int)

    # 4. Blood Pressure Discretization
    if "BloodPressure" in df.columns:
        trans_df["BP_Normal(<80)"] = (df["BloodPressure"] < 80).astype(int)
        trans_df["BP_Elevated(>=80)"] = (df["BloodPressure"] >= 80).astype(int)

    # 5. Outcome Target
    if target_column in df.columns:
        trans_df["Outcome_Diabetic"] = (df[target_column] == 1).astype(int)
        trans_df["Outcome_Healthy"] = (df[target_column] == 0).astype(int)

    # Convert to boolean DataFrame for mlxtend
    return trans_df.astype(bool)


def mine_association_rules(df: pd.DataFrame,
                            min_support: float = 0.08,
                            min_confidence: float = 0.40,
                            target_column: str = "Outcome") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Apriori frequent itemset generation and association rules extraction.
    Returns (frequent_itemsets, rules_df).
    """
    trans_bool = prepare_transactional_data(df, target_column)

    # Generate frequent itemsets
    frequent_itemsets = apriori(trans_bool, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame(), pd.DataFrame()

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return frequent_itemsets, pd.DataFrame()

    # Format antecedents and consequents into readable strings
    rules["Antecedents"] = rules["antecedents"].apply(lambda s: " + ".join(list(s)))
    rules["Consequents"] = rules["consequents"].apply(lambda s: " + ".join(list(s)))
    rules["support"] = rules["support"].round(4)
    rules["confidence"] = rules["confidence"].round(4)
    rules["lift"] = rules["lift"].round(4)
    rules["leverage"] = rules["leverage"].round(4)

    # Sort primarily by lift (strength of association beyond random chance)
    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules


def plot_rules_scatter(rules: pd.DataFrame) -> go.Figure:
    """Plot Support vs Confidence scatter colored by Lift."""
    if rules.empty:
        return go.Figure()

    fig = px.scatter(
        rules,
        x="support",
        y="confidence",
        color="lift",
        color_continuous_scale="Viridis",
        hover_data=["Antecedents", "Consequents"],
        title="Association Rules: Support vs Confidence (Colored by Lift)"
    )
    fig.update_layout(
        xaxis_title="Support (Prevalence in Cohort)",
        yaxis_title="Confidence (Conditional Probability)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter")
    )
    return fig


ASSOCIATION_CAUSATION_DISCLAIMER = r"""
> **Methodological Principle**: Association rules quantify statistical co-occurrence 
> within historical observations ($A \implies B$). A high lift indicates that $A$ and $B$ 
> occur together significantly more often than expected by chance. **Lift does NOT imply 
> medical causality.** Associations highlight candidate screening markers, not direct biological mechanisms.
"""
