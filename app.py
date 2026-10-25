import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Beauty Price Prediction", page_icon="💄", layout="wide")

st.title("💄 Beauty Product Price Prediction")
st.write("Predict a beauty product's selling price using product details from the Nykaa dataset.")
st.caption("Dataset: Nykaa popular brands products (2022). Predictions reflect the patterns in this dataset, not guaranteed current market prices.")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource
def train_models(data, use_mrp=True):
    feature_columns = ["brand_name", "product_title", "mrp", "rating", "rating_count", "in_stock"]
    if not use_mrp:
        feature_columns.remove("mrp")

    target = "price"
    model_data = data[feature_columns + [target]].copy()
    model_data["price"] = pd.to_numeric(model_data["price"], errors="coerce")
    for col in ["mrp", "rating", "rating_count"]:
        if col in model_data:
            model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
    model_data = model_data.dropna(subset=["price"])
    model_data = model_data[model_data["price"] > 0]
    for col in ["brand_name", "product_title", "in_stock"]:
        model_data[col] = model_data[col].fillna("Unknown").astype(str)
    for col in ["rating", "rating_count", "mrp"]:
        if col in model_data:
            model_data[col] = model_data[col].fillna(model_data[col].median())

    X = model_data[feature_columns]
    y = model_data[target]
    num_cols = [c for c in ["mrp", "rating", "rating_count"] if c in feature_columns]
    cat_cols = ["brand_name", "in_stock"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("title_text", TfidfVectorizer(max_features=300, stop_words="english"), "product_title")
        ],
        remainder="drop"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=15, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    }
    fitted = {}
    rows = []
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "MAE (₹)": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "RMSE (₹)": np.sqrt(mean_squared_error(y_test, pred)),
            "R² Score": r2_score(y_test, pred)
        })
    metrics = pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    return fitted, metrics, X_test, y_test, feature_columns, model_data

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Price", "Dataset & EDA", "Model Training & Evaluation", "Statistics Practicals"])

st.sidebar.divider()
st.sidebar.subheader("Dataset")
uploaded = st.sidebar.file_uploader("Upload Nykaa CSV", type=["csv"])
default_path = "nyka_popular_brands_products_2022_10_16.csv"

data = None
if uploaded is not None:
    try:
        data = load_data(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
else:
    try:
        data = pd.read_csv(default_path)
        st.sidebar.caption(f"Loaded local file: {default_path}")
    except FileNotFoundError:
        st.sidebar.info("Upload the Nykaa CSV in the sidebar, or place it beside app.py.")

if data is None:
    st.info("To begin, upload `nyka_popular_brands_products_2022_10_16.csv` using the sidebar.")
    st.stop()

required = {"price", "mrp", "brand_name", "product_title", "rating", "rating_count", "in_stock"}
missing = required - set(data.columns)
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(sorted(missing)))
    st.stop()

if page == "Home":
    st.subheader("Project Overview")
    st.write("""
    This project uses supervised machine learning regression to estimate the selling price of a beauty product.
    It uses product information such as brand, title, MRP, rating, rating count and stock status.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(data):,}")
    c2.metric("Dataset columns", f"{len(data.columns)}")
    c3.metric("Target variable", "price")
    st.markdown("### Workflow")
    st.write("Dataset → Data cleaning → EDA → Preprocessing → Regression models → Evaluation → Price prediction")
    st.warning("The data is from 2022. It is appropriate for an academic project, but its prices may not match today's prices.")
    st.markdown("### Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)

elif page == "Predict Price":
    st.subheader("Estimate a Beauty Product's Selling Price")
    st.write("Enter the product details below. The app trains models on the uploaded dataset and uses the best test-set R² model.")
    use_mrp = st.checkbox("Use MRP as an input feature", value=True,
                          help="If checked, enter the product's MRP. Uncheck if you want an estimate without MRP.")
    with st.spinner("Training and evaluating regression models..."):
        try:
            fitted, metrics, X_test, y_test, feature_columns, model_data = train_models(data, use_mrp)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    st.success(f"Best model on this test split: {best_name}")
    brands = sorted(data["brand_name"].dropna().astype(str).unique().tolist())
    stocks = sorted(data["in_stock"].dropna().astype(str).unique().tolist())
    if not stocks:
        stocks = ["Unknown"]
    with st.form("prediction_form"):
        brand = st.selectbox("Brand", brands)
        title = st.text_input("Product title", "Face Cream")
        if use_mrp:
            mrp = st.number_input("MRP (₹)", min_value=1.0, value=float(max(1, pd.to_numeric(data["mrp"], errors="coerce").median())))
        else:
            mrp = 0.0
        rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=float(np.clip(pd.to_numeric(data["rating"], errors="coerce").median(), 0, 5)), step=0.1)
        rating_count = st.number_input("Rating count", min_value=0, value=int(max(0, pd.to_numeric(data["rating_count"], errors="coerce").median())))
        stock = st.selectbox("In stock", stocks)
        submitted = st.form_submit_button("Predict Price", type="primary")
    if submitted:
        row = {
            "brand_name": brand,
            "product_title": title,
            "rating": rating,
            "rating_count": rating_count,
            "in_stock": stock
        }
        if use_mrp:
            row["mrp"] = mrp
        input_df = pd.DataFrame([row])[feature_columns]
        prediction = float(best_model.predict(input_df)[0])
        prediction = max(0, prediction)
        st.markdown("### Predicted selling price")
        st.metric("Estimated price", f"₹{prediction:,.2f}")
        st.caption("This is a model estimate, not a live Nykaa price or a guaranteed price.")

elif page == "Dataset & EDA":
    st.subheader("Dataset Overview")
    a, b = st.columns(2)
    a.metric("Rows", f"{data.shape[0]:,}")
    b.metric("Columns", f"{data.shape[1]}")
    st.markdown("### Preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown("### Data types")
    st.dataframe(data.dtypes.astype(str).rename("Data type").to_frame(), use_container_width=True)
    st.markdown("### Missing values")
    missing_df = data.isnull().sum().sort_values(ascending=False).rename("Missing count").to_frame()
    st.dataframe(missing_df, use_container_width=True)
    st.markdown("### Duplicate rows")
    st.write(int(data.duplicated().sum()))
    st.markdown("### Descriptive statistics")
    st.dataframe(data.describe(include="all").T, use_container_width=True)
    clean_prices = pd.to_numeric(data["price"], errors="coerce").dropna()
    clean_prices = clean_prices[clean_prices > 0]
    st.markdown("### Selling price distribution")
    fig, ax = plt.subplots()
    sns.histplot(clean_prices, bins=40, kde=True, ax=ax)
    ax.set_xlabel("Selling price (₹)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Price box plot")
    fig, ax = plt.subplots()
    sns.boxplot(x=clean_prices, ax=ax)
    ax.set_xlabel("Selling price (₹)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### MRP vs selling price")
    scatter_df = data[["mrp", "price"]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots()
    sns.scatterplot(data=scatter_df, x="mrp", y="price", alpha=0.5, ax=ax)
    ax.set_xlabel("MRP (₹)")
    ax.set_ylabel("Selling price (₹)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Correlation matrix")
    numeric = data[["price", "mrp", "rating", "rating_count"]].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Top 10 brands")
    top_brands = data["brand_name"].astype(str).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_brands.index, y=top_brands.values, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Product count")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Model Training & Evaluation":
    st.subheader("Regression Model Training and Evaluation")
    with st.spinner("Training regression models..."):
        try:
            fitted, metrics, X_test, y_test, feature_columns, model_data = train_models(data, True)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()
    st.dataframe(metrics, use_container_width=True)
    best_name = metrics.iloc[0]["Model"]
    best_pred = fitted[best_name].predict(X_test)
    st.success(f"Best model by test R²: {best_name}")
    st.markdown("### Actual vs predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, best_pred, alpha=0.6)
    low = min(y_test.min(), best_pred.min())
    high = max(y_test.max(), best_pred.max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual price (₹)")
    ax.set_ylabel("Predicted price (₹)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Residual plot")
    fig, ax = plt.subplots()
    ax.scatter(best_pred, y_test - best_pred, alpha=0.6)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted price (₹)")
    ax.set_ylabel("Residual (actual - predicted)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Download trained model")
    model_bytes = __import__("io").BytesIO()
    joblib.dump(fitted[best_name], model_bytes)
    st.download_button("Download best model (.pkl)", data=model_bytes.getvalue(), file_name="beauty_price_model.pkl")
    st.download_button("Download feature names (.pkl)", data=__import__("pickle").dumps(feature_columns), file_name="beauty_price_features.pkl")

elif page == "Statistics Practicals":
    st.subheader("Statistics on Beauty Product Prices")
    prices = pd.to_numeric(data["price"], errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) < 2:
        st.warning("Not enough valid prices for statistics.")
        st.stop()
    q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
    stats_table = pd.Series({
        "Mean": prices.mean(), "Median": prices.median(), "Mode": prices.mode().iloc[0],
        "Minimum": prices.min(), "Maximum": prices.max(), "Range": prices.max()-prices.min(),
        "Variance": prices.var(), "Standard deviation": prices.std(),
        "Q1": q1, "Q3": q3, "IQR": q3-q1
    })
    st.markdown("### Descriptive statistics")
    st.dataframe(stats_table.rename("Value").to_frame())
    st.markdown("### Empirical probabilities")
    st.write(f"P(price ≤ ₹500) = {(prices <= 500).mean():.4f}")
    st.write(f"P(price > ₹1000) = {(prices > 1000).mean():.4f}")
    st.write(f"P(₹500 ≤ price ≤ ₹1500) = {prices.between(500, 1500).mean():.4f}")
    st.markdown("### Sampling distribution of the mean")
    np.random.seed(42)
    population = prices.to_numpy()
    n = min(30, len(population))
    means = [np.random.choice(population, size=n, replace=True).mean() for _ in range(1000)]
    fig, ax = plt.subplots()
    ax.hist(means, bins=30, edgecolor="black")
    ax.set_xlabel("Sample mean price (₹)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### 95% confidence interval for mean price")
    mean_price = prices.mean()
    std_price = prices.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=len(prices)-1)
    margin = t_crit * std_price / np.sqrt(len(prices))
    st.write(f"Mean: ₹{mean_price:,.2f}")
    st.write(f"95% confidence interval: ₹{mean_price-margin:,.2f} to ₹{mean_price+margin:,.2f}")
    st.markdown("### One-sample t-test")
    reference = st.number_input("Reference mean price (₹)", min_value=0.0, value=500.0, step=50.0)
    t_stat, p_val = stats.ttest_1samp(prices, popmean=reference)
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_val:.6g}")
    st.write("Reject the null hypothesis at 5% significance." if p_val < 0.05 else "Fail to reject the null hypothesis at 5% significance.")
    st.markdown("### Paired t-test: MRP vs selling price")
    paired = data[["mrp", "price"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(paired) > 1:
        paired = paired[(paired["mrp"] > 0) & (paired["price"] > 0)]
        t2, p2 = stats.ttest_rel(paired["mrp"], paired["price"])
        st.write(f"Mean MRP: ₹{paired['mrp'].mean():,.2f}")
        st.write(f"Mean selling price: ₹{paired['price'].mean():,.2f}")
        st.write(f"p-value: {p2:.6g}")
    st.markdown("### One-way ANOVA across top five brands")
    top_five = data["brand_name"].value_counts().head(5).index
    groups = [pd.to_numeric(data.loc[data["brand_name"] == brand, "price"], errors="coerce").dropna() for brand in top_five]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f_stat, p3 = stats.f_oneway(*groups)
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"p-value: {p3:.6g}")
    else:
        st.info("Not enough brand groups to perform ANOVA.")
    st.caption("These tests are educational analyses. Interpret them with their assumptions and the chosen reference values in mind.")
