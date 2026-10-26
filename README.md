# Beauty-Price-Prediction-System
BEAUTY PRODUCT PRICE PREDICTION - STREAMLIT APP

FILES REQUIRED
- app.py
- requirements.txt
- nyka_popular_brands_products_2022_10_16.csv

SETUP (Windows)
1. Put all three files in the same folder.
2. Open Command Prompt/Terminal in that folder.
3. Create a virtual environment (recommended):
   py -m venv .venv
   .venv\Scripts\activate
4. Install libraries:
   python -m pip install -r requirements.txt
5. Run:
   streamlit run app.py
6. If the browser does not open automatically, open the Local URL shown in terminal.

USING THE APP
- Upload the CSV in the left sidebar, or keep the CSV next to app.py.
- Home: project overview and dataset preview.
- Predict Price: enter product attributes and predict selling price.
- Dataset & EDA: inspect missing values, summary statistics and plots.
- Model Training & Evaluation: compare regression models and download the best model.
- Statistics Practicals: descriptive statistics, probability, sampling distribution, confidence interval and selected hypothesis tests.

NOTES
- The dataset is from 2022, so prices may not reflect current prices.
- The app trains models locally when needed. Training can take a little time.
- Predictions are estimates, not live Nykaa prices.
