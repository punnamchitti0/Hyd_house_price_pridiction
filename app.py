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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Keep this CSV file in the same folder as app.py.
DATASET_PATH = Path(__file__).parent / "Hyderbad_House_price.csv"


def identify_columns(df):
    """Identify the price, area, and location columns in the uploaded dataset."""
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    target = next(
        (c for c in cols if "price" in lower[c] and "sqft" not in lower[c]),
        None
    )

    # Prefer the actual property area column over rate-per-square-foot.
    area = next((c for c in cols if "area" in lower[c]), None)
    if area is None:
        area = next(
            (c for c in cols if any(x in lower[c] for x in ["sqft", "size"])),
            None
        )

    location = next(
        (c for c in cols if any(
            x in lower[c] for x in ["location", "locality", "city", "loc"]
        )),
        None
    )
    bedrooms = next(
        (c for c in cols if "bed" in lower[c] or "bhk" in lower[c]),
        None
    )
    bathrooms = next((c for c in cols if "bath" in lower[c]), None)

    if target is None:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        target = numeric[-1] if numeric else None

    return {
        "target": target,
        "area": area,
        "location": location,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }


def prepare_data(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Remove CSV index columns such as "Unnamed: 0".
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    mapping = identify_columns(df)
    target = mapping["target"]

    if target is None:
        raise ValueError(
            "Could not identify the house-price column. "
            "Please include a column named Price or price(L)."
        )

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.drop_duplicates()
    df = df[df[target].notna() & (df[target] > 0)]

    if mapping["area"]:
        df[mapping["area"]] = pd.to_numeric(
            df[mapping["area"]], errors="coerce"
        )
        df = df[df[mapping["area"]].isna() | (df[mapping["area"]] > 100)]

    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check your CSV.")

    # Fill missing numeric and categorical values.
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(exclude=np.number).columns:
        modes = df[col].mode()
        fill_value = modes.iloc[0] if not modes.empty else "Unknown"
        df[col] = df[col].fillna(fill_value).astype(str).str.strip()

    return df, mapping


@st.cache_resource
def train_model(data_json, target):
    df = pd.read_json(io.StringIO(data_json), orient="records")
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=180,
            random_state=42,
            max_depth=12,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R² score": r2_score(y_test, predictions),
        "MAE (₹ lakh)": mean_absolute_error(y_test, predictions),
        "RMSE (₹ lakh)": np.sqrt(mean_squared_error(y_test, predictions))
    }

    return model, features, metrics


st.title("🏠 Hyderabad House Price Prediction")
st.caption(
    "Explore the Hyderabad housing dataset, understand market patterns, "
    "and estimate a property's price."
)

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader(
        "Upload another house-price CSV (optional)",
        type=["csv"]
    )
    st.caption(
        "The included Hyderbad_House_price.csv file is loaded automatically. "
        "Upload another CSV only if you want to replace it."
    )
    st.divider()
    st.write("**Project workflow**")
    st.write("1. Load Hyderabad house-price CSV")
    st.write("2. Clean the data")
    st.write("3. Explore housing patterns")
    st.write("4. Train a Random Forest model")
    st.write("5. Predict house prices")

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.sidebar.success("Using the CSV you uploaded.")
    elif DATASET_PATH.exists():
        raw = pd.read_csv(DATASET_PATH)
        st.sidebar.success("Hyderbad_House_price.csv loaded successfully.")
    else:
        st.error(
            "Dataset file not found. Keep Hyderbad_House_price.csv "
            "in the same folder as app.py."
        )
        st.stop()

    df, mapping = prepare_data(raw)
    target = mapping["target"]

    # The supplied dataset's price(L) column stores prices in lakh rupees.
    price_is_lakh = "l" in str(target).lower() or "lakh" in str(target).lower()
    price_unit = "₹ lakh" if price_is_lakh else "₹"

    model, features, metrics = train_model(
        df.to_json(orient="records"),
        target
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🏡 Predict Price",
        "📁 Dataset"
    ])

    with tab1:
        st.subheader("Market overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valid properties", f"{len(df):,}")

        if price_is_lakh:
            c2.metric("Average price", f"₹{df[target].mean():,.2f} L")
            c3.metric("Median price", f"₹{df[target].median():,.2f} L")
        else:
            c2.metric("Average price", f"₹{df[target].mean():,.0f}")
            c3.metric("Median price", f"₹{df[target].median():,.0f}")

        c4.metric("Model R² score", f"{metrics['R² score']:.3f}")

        left, right = st.columns(2)

        with left:
            st.markdown("**House price distribution**")
            fig, ax = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax)
            ax.set_xlabel(f"Price ({price_unit})")
            ax.set_ylabel("Number of properties")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Average price by location**")
            if mapping["location"]:
                loc = (
                    df.groupby(mapping["location"])[target]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                )
                fig, ax = plt.subplots()
                loc.plot(kind="bar", ax=ax)
                ax.set_ylabel(f"Average price ({price_unit})")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No location column was detected in the CSV.")

        st.markdown("**Model evaluation**")
        metric_display = pd.DataFrame([metrics]).style.format({
            "R² score": "{:.3f}",
            "MAE (₹ lakh)": "₹{:.2f} lakh",
            "RMSE (₹ lakh)": "₹{:.2f} lakh"
        })
        st.dataframe(metric_display, use_container_width=True)
        st.caption(
            "R², MAE and RMSE are calculated using a held-out test set. "
            "The results depend on the supplied dataset."
        )

    with tab2:
        st.subheader("Estimate a house price")
        st.write(
            "Enter the property details. The model uses the columns "
            "available in your Hyderabad dataset."
        )

        values = {}
        cols = st.columns(2)

        for i, col in enumerate(features):
            with cols[i % 2]:
                series = df[col]

                if series.dtype == "object":
                    options = sorted(series.astype(str).unique().tolist())
                    values[col] = st.selectbox(
                        col,
                        options,
                        key=f"pred_{col}"
                    )
                elif (
                    pd.api.types.is_integer_dtype(series)
                    or series.nunique() <= 10
                ):
                    low = int(series.min())
                    high = int(series.max())
                    values[col] = st.number_input(
                        col,
                        min_value=low,
                        max_value=max(low, high),
                        value=int(series.median()),
                        step=1,
                        key=f"pred_{col}"
                    )
                else:
                    low = float(series.min())
                    high = float(series.max())
                    values[col] = st.number_input(
                        col,
                        min_value=float(max(0, low)),
                        max_value=float(max(high, low + 1)),
                        value=float(series.median()),
                        key=f"pred_{col}"
                    )

        if st.button(
            "Predict House Price",
            type="primary",
            use_container_width=True
        ):
            prediction = float(
                model.predict(pd.DataFrame([values], columns=features))[0]
            )

            if price_is_lakh:
                st.success(f"Estimated house price: ₹{prediction:,.2f} lakh")
                st.caption(
                    f"Approximately ₹{prediction * 100000:,.0f}"
                )
            else:
                st.success(f"Estimated house price: ₹{prediction:,.0f}")

            st.caption(
                "This is a machine-learning estimate, not a professional "
                "property valuation."
            )

    with tab3:
        st.subheader("Cleaned dataset preview")
        st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
        st.dataframe(df.head(100), use_container_width=True)

        cleaned_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download cleaned CSV",
            cleaned_csv,
            file_name="Hyderabad_House_Price_Cleaned.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Unable to load or process this dataset: {e}")
    st.info(
        "Check that your CSV contains a numeric house-price column "
        "(such as price(L)) and property details."
    )
