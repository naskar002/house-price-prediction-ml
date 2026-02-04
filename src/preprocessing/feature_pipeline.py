from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline for house price data.

    This transformer accepts a pandas DataFrame and returns a transformed DataFrame.

    Steps implemented (in order):
    - Create `House_Age` from `Year Built` and optionally drop `Year Built`.
    - Impute numeric/categorical columns using medians/modes computed on fit data.
    - Convert `Date Sold` -> `Sold_Year`, `Sold_Month` and drop `Date Sold`.
    - Ordinal-encode `Condition` using a fixed mapping.
    - One-hot encode `Type` and `Location` (drop_first configurable).
    - Drop identifier columns (e.g., `Property ID`).

    Use `fit` on training data to learn medians/modes and allowed categories,
    then call `transform` on unseen data.
    """

    def __init__(self,
                 current_year: int = 2026,
                 drop_first_dummies: bool = True,
                 drop_sold_month: bool = True,
                 drop_sold_year: bool = True,
                 drop_property_id: bool = True):
        self.current_year = current_year
        self.drop_first_dummies = drop_first_dummies
        self.drop_sold_month = drop_sold_month
        self.drop_sold_year = drop_sold_year
        self.drop_property_id = drop_property_id

        # learned attributes
        self.medians_ = {}
        self.modes_ = {}
        self.type_categories_: List[str] = []
        self.location_categories_: List[str] = []

        # fixed ordinal mapping for Condition
        self.condition_order = {
            "Poor": 1,
            "Fair": 2,
            "Good": 3,
            "New": 4,
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        df = X.copy()

        # 1. Create House_Age during fit (so median can be learned)
        if "Year Built" in df.columns:
            df["House_Age"] = self.current_year - df["Year Built"]

        # 2. Learn medians (Size, House_Age)
        median_cols = ["Size", "House_Age"]
        for c in median_cols:
            if c in df.columns:
                self.medians_[c] = df[c].median()

        # 3. Learn modes for numeric discrete (Bedrooms, Bathrooms)
        mode_numeric_cols = ["Bedrooms", "Bathrooms"]
        for c in mode_numeric_cols:
            if c in df.columns:
                mode = df[c].mode()
                self.modes_[c] = mode.iloc[0] if not mode.empty else np.nan

        # 4. Learn modes for categorical columns
        cat_cols = ["Condition", "Type", "Location"]
        for c in cat_cols:
            if c in df.columns:
                mode = df[c].mode()
                self.modes_[c] = mode.iloc[0] if not mode.empty else np.nan

        # 5. Learn Type categories (order preserved)
        if "Type" in df.columns:
            self.type_categories_ = [
                str(x) for x in pd.Series(df["Type"].dropna().unique()).tolist()
            ]

        # 6. Learn Location categories (order preserved)
        if "Location" in df.columns:
            self.location_categories_ = [
                str(x) for x in pd.Series(df["Location"].dropna().unique()).tolist()
            ]

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. House_Age
        if "Year Built" in df.columns:
            df["House_Age"] = self.current_year - df["Year Built"]
            df.drop(columns=["Year Built"], inplace=True)

        # 2. Imputation
        # numeric imputation using medians
        for c, m in self.medians_.items():
            if c in df.columns:
                df[c] = df[c].fillna(m)

        # categorical imputation using modes
        for c, m in self.modes_.items():
            if c in df.columns:
                df[c] = df[c].fillna(m)

        # 3. Date features
        if "Date Sold" in df.columns:
            try:
                df["Date Sold"] = pd.to_datetime(df["Date Sold"], errors="coerce")
                df["Sold_Year"] = df["Date Sold"].dt.year
                df["Sold_Month"] = df["Date Sold"].dt.month
            except Exception:
                df["Sold_Year"] = np.nan
                df["Sold_Month"] = np.nan
            # drop Date Sold
            df.drop(columns=["Date Sold"], inplace=True)
            if self.drop_sold_month and "Sold_Month" in df.columns:
                df.drop(columns=["Sold_Month"], inplace=True)

        # 4. Drop Property ID
        if self.drop_property_id and "Property ID" in df.columns:
            df.drop(columns=["Property ID"], inplace=True)

        # 5. Ordinal encode Condition
        if "Condition" in df.columns:
            df["Condition"] = df["Condition"].map(self.condition_order)


        # 6. One-hot encode Type
        if "Type" in df.columns:
            categories = self.type_categories_ if self.type_categories_ else [str(x) for x in df["Type"].dropna().unique()]
            # ensure categories are strings
            categories = [str(c) for c in categories]
            # optionally drop first
            start_idx = 1 if self.drop_first_dummies and len(categories) > 0 else 0
            expected = [f"Type_{cat}" for cat in categories[start_idx:]]

            dummies = pd.get_dummies(df["Type"].astype(str), prefix="Type")
            # if drop_first, remove the first category column if present
            if self.drop_first_dummies and categories:
                first_col = f"Type_{categories[0]}"
                if first_col in dummies.columns:
                    dummies = dummies.drop(columns=[first_col])

            # ensure all expected columns exist
            for col in expected:
                if col not in dummies.columns:
                    dummies[col] = 0

            # align order
            dummies = dummies[expected]
            df = pd.concat([df.drop(columns=["Type"]), dummies], axis=1)

        # 7. One-hot encode Location
        if "Location" in df.columns:
            categories = self.location_categories_ if self.location_categories_ else [str(x) for x in df["Location"].dropna().unique()]
            categories = [str(c) for c in categories]
            start_idx = 1 if self.drop_first_dummies and len(categories) > 0 else 0
            expected = [f"Location_{cat}" for cat in categories[start_idx:]]

            dummies = pd.get_dummies(df["Location"].astype(str), prefix="Location")
            if self.drop_first_dummies and categories:
                first_col = f"Location_{categories[0]}"
                if first_col in dummies.columns:
                    dummies = dummies.drop(columns=[first_col])

            for col in expected:
                if col not in dummies.columns:
                    dummies[col] = 0

            dummies = dummies[expected]
            df = pd.concat([df.drop(columns=["Location"]), dummies], axis=1)

        # 8. Optionally drop Sold_Year if requested
        if self.drop_sold_year and "Sold_Year" in df.columns:
            df.drop(columns=["Sold_Year"], inplace=True)

        return df

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)


def build_default_pipeline() -> FeatureEngineeringPipeline:
    """Convenience factory for the default pipeline used in EDA notebook."""
    return FeatureEngineeringPipeline(current_year=2026,
                                      drop_first_dummies=True,
                                      drop_sold_month=True,
                                      drop_sold_year=True,
                                      drop_property_id=True)
