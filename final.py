# ✅ Adzuna Job Scraper – Collecting, Cleaning, Modeling, and Saving Job Data
import requests
import pandas as pd
from google.colab import files  # For downloading the CSV in Colab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# --- API credentials ---
app_id = "35e20e56"
app_key = "f258a757c25b06015762323b593d49a4"

# --- Fetch jobs from Adzuna across North America (Canada + US) ---
def fetch_adzuna_jobs(role="data analyst", countries=["ca", "us"], results=8000):
    all_jobs = []

    for country in countries:
        pages = results // 50  # Adzuna returns up to 50 results per page
        for page in range(1, pages + 1):
            base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
            params = {
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": 50,
                "what": role,
                "content-type": "application/json"
            }

            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching from {country.upper()}, page {page}: {response.status_code}")
                continue

            data = response.json()
            for job in data.get("results", []):
                all_jobs.append({
                    "title": job.get("title"),
                    "company": job.get("company", {}).get("display_name"),
                    "location": job.get("location", {}).get("display_name"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "contract_time": job.get("contract_time"),
                    "description": job.get("description")[:100],
                    "url": job.get("redirect_url")
                })

    # Check for duplicates and missing company names
    temp_df = pd.DataFrame(all_jobs)
    duplicate_count = temp_df.duplicated(subset=["title", "company", "location", "url"]).sum()
    missing_companies = temp_df["company"].isna().sum()
    print(f"🔍 Duplicate listings: {duplicate_count}")
    print(f"🛑 Listings missing company name: {missing_companies}")

    temp_df = temp_df.drop_duplicates(subset=["title", "company", "location", "url"])
    print(f"✅ Duplicates removed. Remaining records: {len(temp_df)}")
    return temp_df

# --- Clean and prepare data ---
def clean_job_data(df, df_raw):
    print(f"Total jobs fetched: {len(df)}")

    df = df.dropna(subset=["salary_min", "salary_max"])
    print(f"Jobs with valid salary data: {len(df)}")

    if len(df) < 5:
        print("Too few records with salary info. Sample of raw data:")
        print(df_raw[["title", "salary_min", "salary_max"]].head(10))

    df["avg_salary"] = (df["salary_min"] + df["salary_max"]) / 2
    df["title"] = df["title"].str.lower()
    df["is_senior"] = df["title"].apply(lambda x: 1 if "senior" in x else 0)
    df["is_junior"] = df["title"].apply(lambda x: 1 if "junior" in x or "entry" in x else 0)
    df["location_code"] = df["location"].astype("category").cat.codes

    return df

# --- Train and evaluate model ---
def train_salary_model(df):
    X = df[["is_senior", "is_junior", "location_code"]]
    y = df["avg_salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict full dataset
    full_predictions = model.predict(X)
    df["predicted_salary"] = full_predictions

    # Evaluate on test set
    test_predictions = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, test_predictions)
    avg_salary_test = y_test.mean()
    test_accuracy = 100 - (mae_test / avg_salary_test * 100)

    print(f"\n✅ Test Set Accuracy: {test_accuracy:.2f}%")

    # Evaluate on full dataset
    mae_full = mean_absolute_error(y, full_predictions)
    avg_salary_full = y.mean()
    full_accuracy = 100 - (mae_full / avg_salary_full * 100)
    print(f"✅ Full Dataset Accuracy: {full_accuracy:.2f}%")

    # Save entire dataset with predictions
    df.to_csv("full_salary_predictions.csv", index=False)
    print("✅ Full dataset with predictions saved to 'full_salary_predictions.csv'")
    files.download("full_salary_predictions.csv")

    return model

# --- Run collection, cleaning, and modeling pipeline ---
df_raw = fetch_adzuna_jobs("data analyst", countries=["ca", "us"], results=8000)

if df_raw.empty or df_raw[["salary_min", "salary_max"]].dropna().empty:
    print("No jobs with salary data returned by API. Please try a different role, location, or increase the result count.")
else:
    df_cleaned = clean_job_data(df_raw.copy(), df_raw)

    if not df_cleaned.empty:
        df_cleaned.to_csv("cleaned_job_data.csv", index=False)
        print("✅ Cleaned dataset saved to 'cleaned_job_data.csv'")
        files.download("cleaned_job_data.csv")

        # Train model and generate predictions for entire dataset
        train_salary_model(df_cleaned)
    else:
        print("Not enough salary data to save a dataset.")
