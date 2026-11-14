
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hyderabad House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

DATASET_PATH = Path(__file__).parent / "Hyderbad_House_price.csv"


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data(file_bytes=None):

    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    elif DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)

    else:
        raise FileNotFoundError(
            "Hyderbad_House_price.csv not found. "
            "Place it in the same folder as app.py."
        )

    return df


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

def prepare_data(raw):

    df = raw.copy()

    df.columns = df.columns.astype(str).str.strip()

    # Remove unnecessary index column
    df = df.loc[
        :,
        ~df.columns.str.contains("^Unnamed", case=False)
    ]

    # Explicitly identify the target column
    if "price(L)" not in df.columns:

        raise ValueError(
            "The CSV must contain a price(L) column."
        )

    target = "price(L)"

    # Numerical columns
    numeric_columns = [
        "price(L)",
        "rate_persqft",
        "area_insqft"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )

    # Remove invalid prices
    df = df.drop_duplicates()

    df = df[
        df["price(L)"].notna() &
        (df["price(L)"] > 0)
    ]

    if "area_insqft" in df.columns:

        df = df[
            df["area_insqft"].isna() |
            (df["area_insqft"] > 100)
        ]

    if len(df) < 10:

        raise ValueError(
            "Not enough valid records to train the model."
        )

    # Fill missing numerical values
    for col in df.select_dtypes(include=np.number).columns:

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Fill missing categorical values
    for col in df.select_dtypes(exclude=np.number).columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    return df, target


# --------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(data_json):

    df = pd.read_json(
        io.StringIO(data_json),
        orient="records"
    )

    target = "price(L)"

    # Separate input and output
    X = df.drop(columns=[target])
    y = df[target]

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Random Forest model
    model = Pipeline([
        ("preprocessor", preprocessor),

        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Evaluation
    metrics = {
        "R2 Score": r2_score(y_test, predictions),

        "MAE": mean_absolute_error(
            y_test, predictions
        ),

        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        )
    }

    return model, X.columns.tolist(), metrics


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.title("🏠 Hyderabad House Price Prediction")

st.markdown(
    """
    ### Machine Learning Based House Price Estimator

    Analyze Hyderabad property prices and predict
    estimated house prices using Random Forest Regression.
    """
)

st.divider()


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload another CSV (optional)",
        type=["csv"]
    )

    st.info(
        "Your Hyderabad CSV is loaded automatically. "
        "Upload another file only if you want to replace it."
    )

    st.divider()

    st.subheader("Project Workflow")

    st.write("1. Dataset Loading")
    st.write("2. Data Cleaning")
    st.write("3. Exploratory Data Analysis")
    st.write("4. Machine Learning")
    st.write("5. House Price Prediction")


# --------------------------------------------------
# 7. MAIN APPLICATION
# --------------------------------------------------

try:

    file_bytes = (
        uploaded_file.getvalue()
        if uploaded_file is not None
        else None
    )

    raw = load_data(file_bytes)

    df, target = prepare_data(raw)

    st.success(
        f"Dataset successfully loaded! "
        f"{len(df):,} valid property records found."
    )

    # Train model
    model, features, metrics = train_model(
        df.to_json(orient="records")
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🏡 Predict Price",
        "📁 Dataset"
    ])


    # --------------------------------------------------
    # TAB 1: DASHBOARD
    # --------------------------------------------------

    with tab1:

        st.header("📊 Housing Market Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Properties",
            f"{len(df):,}"
        )

        c2.metric(
            "Average Price",
            f"₹{df[target].mean():.2f} L"
        )

        c3.metric(
            "Median Price",
            f"₹{df[target].median():.2f} L"
        )

        c4.metric(
            "Model R² Score",
            f"{metrics['R2 Score']:.3f}"
        )

        st.divider()

        left, right = st.columns(2)

        # Price distribution
        with left:

            st.subheader("House Price Distribution")

            fig, ax = plt.subplots()

            sns.histplot(
                df[target],
                kde=True,
                ax=ax
            )

            ax.set_xlabel("Price (₹ Lakhs)")
            ax.set_ylabel("Number of Properties")

            st.pyplot(fig)

            plt.close(fig)

        # Location analysis
        with right:

            st.subheader("Average Price by Location")

            if "location" in df.columns:

                location_prices = (
                    df.groupby("location")[target]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                )

                fig, ax = plt.subplots()

                location_prices.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylabel("Average Price (₹ Lakhs)")
                ax.set_xlabel("Location")

                plt.xticks(rotation=45)

                st.pyplot(fig)

                plt.close(fig)

        st.divider()

        # Model evaluation
        st.subheader("🤖 Model Evaluation")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "R² Score",
            f"{metrics['R2 Score']:.3f}"
        )

        m2.metric(
            "MAE",
            f"₹{metrics['MAE']:.2f} L"
        )

        m3.metric(
            "RMSE",
            f"₹{metrics['RMSE']:.2f} L"
        )

        st.caption(
            "Evaluation metrics are calculated using "
            "a separate test dataset."
        )


    # --------------------------------------------------
    # TAB 2: HOUSE PRICE PREDICTION
    # --------------------------------------------------

    with tab2:

        st.header("🏡 Predict House Price")

        st.write(
            "Enter the property details to estimate "
            "its market price."
        )

        values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):

            with (col1 if i % 2 == 0 else col2):

                series = df[feature]

                # Categorical features
                if series.dtype == "object":

                    options = sorted(
                        series.unique().tolist()
                    )

                    values[feature] = st.selectbox(
                        feature,
                        options,
                        key=f"input_{feature}"
                    )

                # Numerical features
                else:

                    low = float(series.min())
                    high = float(series.max())

                    if low == high:

                        values[feature] = st.number_input(
                            feature,
                            value=low,
                            disabled=True
                        )

                    else:

                        values[feature] = st.number_input(
                            feature,
                            min_value=low,
                            max_value=high,
                            value=float(series.median()),
                            key=f"input_{feature}"
                        )

        st.divider()

        if st.button(
            "🔍 Predict House Price",
            type="primary",
            use_container_width=True
        ):

            input_df = pd.DataFrame(
                [values],
                columns=features
            )

            prediction = float(
                model.predict(input_df)[0]
            )

            st.success(
                f"🏠 Estimated House Price: ₹{prediction:.2f} Lakhs"
            )

            st.info(
                f"Approximate price in rupees: "
                f"₹{prediction * 100000:,.0f}"
            )

            st.caption(
                "This prediction is an estimate based on "
                "historical dataset patterns, not a professional valuation."
            )


    # --------------------------------------------------
    # TAB 3: DATASET
    # --------------------------------------------------

    with tab3:

        st.header("📁 Hyderabad Housing Dataset")

        st.write(
            f"Total Rows: {len(df):,} | "
            f"Total Columns: {len(df.columns)}"
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Column Information")

        information = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            information,
            use_container_width=True
        )

        # Download cleaned dataset
        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            data=csv_data,
            file_name="Hyderabad_House_Price_Cleaned.csv",
            mime="text/csv"
        )


except Exception as e:

    st.error(
        f"Unable to load or process dataset: {e}"
    )

    st.info(
        "Make sure your CSV contains price(L), "
        "location, title, area_insqft and other "
        "property details."
    )
