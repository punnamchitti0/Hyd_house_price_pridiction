
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


# ==================================================
# 1. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Hyderabad House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

DATASET_PATH = Path(__file__).parent / "Hyderbad_House_price.csv"


# ==================================================
# 2. LOAD DATASET
# ==================================================

@st.cache_data
def load_data(file_bytes=None):

    if file_bytes is not None:

        raw = pd.read_csv(
            io.BytesIO(file_bytes),
            low_memory=False
        )

    else:

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                "Hyderbad_House_price.csv not found. "
                "Keep the CSV in the same folder as app.py."
            )

        raw = pd.read_csv(
            DATASET_PATH,
            low_memory=False
        )

    return raw


# ==================================================
# 3. DATA CLEANING AND COLUMN IDENTIFICATION
# ==================================================

def prepare_data(raw):

    df = raw.copy()

    # Clean column names
    df.columns = [
        str(col).replace("\ufeff", "").strip()
        for col in df.columns
    ]

    # Remove unnecessary index columns
    df = df.loc[
        :,
        ~df.columns.str.contains(
            r"^Unnamed",
            case=False,
            regex=True
        )
    ]

    # Find columns without depending on capitalization
    column_lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # Identify price column
    target = column_lookup.get("price(l)")

    if target is None:
        target = next(
            (
                col for name, col in column_lookup.items()
                if "price" in name and "sqft" not in name
            ),
            None
        )

    if target is None:
        raise ValueError(
            "Price column not found. "
            "Your CSV must contain price(L)."
        )

    # Explicitly identify the supplied dataset columns
    location_col = column_lookup.get("location")
    title_col = column_lookup.get("title")
    area_col = column_lookup.get("area_insqft")
    rate_col = column_lookup.get("rate_persqft")
    status_col = column_lookup.get("building_status")

    # IMPORTANT:
    # These columns must always be treated as text.
    categorical_columns = [
        col for col in [
            location_col,
            title_col,
            status_col
        ]
        if col is not None
    ]

    for col in categorical_columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # IMPORTANT:
    # Only these columns are converted to numbers.
    numeric_columns = [
        target,
        area_col,
        rate_col
    ]

    numeric_columns = [
        col for col in numeric_columns
        if col is not None
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove invalid house prices
    df = df[
        df[target].notna() &
        (df[target] > 0)
    ]

    # Remove invalid area values
    if area_col is not None:

        df = df[
            df[area_col].isna() |
            (df[area_col] > 100)
        ]

    if len(df) < 10:

        raise ValueError(
            "Not enough valid property records to train the model."
        )

    # Fill missing numerical values
    for col in df.select_dtypes(
        include=np.number
    ).columns:

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Fill missing categorical values
    for col in df.select_dtypes(
        exclude=np.number
    ).columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    return df, target, categorical_columns


# ==================================================
# 4. RANDOM FOREST MODEL
# ==================================================

@st.cache_resource
def train_model(df, target):

    # Separate input features and target
    X = df.drop(columns=[target]).copy()

    y = pd.to_numeric(
        df[target],
        errors="raise"
    )

    # Explicitly identify numerical columns
    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Explicitly identify categorical columns
    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Make sure text columns remain text
    for col in categorical_features:

        X[col] = X[col].astype(str)

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    # Combine preprocessing
    transformers = []

    if numerical_features:

        transformers.append((
            "numeric",
            numerical_pipeline,
            numerical_features
        ))

    if categorical_features:

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_features
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Random Forest Regression
    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # Training and testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Model predictions
    predictions = model.predict(X_test)

    # Evaluation metrics
    metrics = {
        "R2 Score": r2_score(
            y_test,
            predictions
        ),

        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    }

    return model, X.columns.tolist(), metrics


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.title(
    "🏠 Hyderabad House Price Prediction"
)

st.markdown(
    """
    ### Machine Learning Based House Price Estimator

    Explore Hyderabad property prices, analyze housing
    trends, and predict estimated property prices using
    Random Forest Regression.
    """
)

st.divider()


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload another CSV (optional)",
        type=["csv"]
    )

    st.caption(
        "The Hyderabad CSV in your project folder "
        "loads automatically."
    )

    st.divider()

    st.subheader("Project Workflow")

    st.write("1. Dataset Loading")
    st.write("2. Data Cleaning")
    st.write("3. Exploratory Data Analysis")
    st.write("4. Random Forest Model")
    st.write("5. House Price Prediction")


# ==================================================
# 7. MAIN APPLICATION
# ==================================================

try:

    # Load uploaded CSV or project CSV
    file_bytes = (
        uploaded_file.getvalue()
        if uploaded_file is not None
        else None
    )

    raw = load_data(file_bytes)

    # Prepare dataset
    df, target, categorical_columns = prepare_data(raw)

    # Show successful dataset loading
    st.success(
        f"Dataset loaded successfully! "
        f"{len(df):,} valid property records found."
    )

    # Show the actual data types for verification
    with st.expander("View dataset column types"):

        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values
            }),
            use_container_width=True
        )

    # Train machine learning model
    model, features, metrics = train_model(
        df,
        target
    )

    # Create application tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🏡 Predict Price",
        "📁 Dataset"
    ])


    # ==================================================
    # TAB 1: DASHBOARD
    # ==================================================

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

        # House price distribution
        with left:

            st.subheader(
                "House Price Distribution"
            )

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

        # Average price by location
        with right:

            st.subheader(
                "Average Price by Location"
            )

            location_col = next(
                (
                    col for col in df.columns
                    if col.lower() == "location"
                ),
                None
            )

            if location_col:

                location_prices = (
                    df.groupby(location_col)[target]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                )

                fig, ax = plt.subplots()

                location_prices.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylabel(
                    "Average Price (₹ Lakhs)"
                )

                ax.set_xlabel("Location")

                plt.xticks(rotation=45)

                st.pyplot(fig)

                plt.close(fig)

            else:

                st.warning(
                    "Location column not found."
                )

        st.divider()

        st.subheader(
            "🤖 Model Evaluation"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "R² Score",
            f"{metrics['R2 Score']:.3f}"
        )

        m2.metric(
            "Mean Absolute Error",
            f"₹{metrics['MAE']:.2f} L"
        )

        m3.metric(
            "Root Mean Squared Error",
            f"₹{metrics['RMSE']:.2f} L"
        )

        st.caption(
            "Model evaluation is performed using "
            "a separate 20% test dataset."
        )


    # ==================================================
    # TAB 2: PREDICT HOUSE PRICE
    # ==================================================

    with tab2:

        st.header(
            "🏡 Predict House Price"
        )

        st.write(
            "Enter the property details to estimate "
            "the house price."
        )

        values = {}

        col1, col2 = st.columns(2)

        for i, feature in enumerate(features):

            with (col1 if i % 2 == 0 else col2):

                series = df[feature]

                # Text / categorical inputs
                if feature in categorical_columns:

                    options = sorted(
                        series.astype(str).unique().tolist()
                    )

                    values[feature] = st.selectbox(
                        feature,
                        options,
                        key=f"input_{feature}"
                    )

                # Numerical inputs
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
                f"🏠 Estimated House Price: "
                f"₹{prediction:.2f} Lakhs"
            )

            st.info(
                f"Approximate price in rupees: "
                f"₹{prediction * 100000:,.0f}"
            )

            st.caption(
                "This is an estimated price based on "
                "historical property data."
            )


    # ==================================================
    # TAB 3: DATASET
    # ==================================================

    with tab3:

        st.header(
            "📁 Hyderabad Housing Dataset"
        )

        st.write(
            f"Total Rows: {len(df):,} | "
            f"Total Columns: {len(df.columns)}"
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader(
            "Dataset Information"
        )

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


# ==================================================
# 8. ERROR HANDLING
# ==================================================

except Exception as e:

    st.error(
        f"Unable to load or process dataset: {e}"
    )

    st.subheader(
        "Debugging Information"
    )

    st.exception(e)

    st.info(
        "Check that your CSV contains price(L), "
        "title, location, rate_persqft, "
        "area_insqft and building_status."
    )
