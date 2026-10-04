import csv
import numpy as np
import pandas as pd
import analytics_module
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


FILE = "social_media_performance.csv"


# ============================================================
# A. LOAD DATASET
# ============================================================

def load_data():

    try:
        df = pd.read_csv(FILE)

        if df.empty:
            print("\nERROR: Dataset is empty.")
            return None

        required = [
            "post_id",
            "platform",
            "content_type",
            "topic",
            "language",
            "region",
            "post_datetime",
            "hashtags",
            "sentiment_score",
            "views",
            "likes",
            "comments",
            "shares",
            "engagement_rate",
            "is_viral"
        ]

        missing = [col for col in required if col not in df.columns]

        if missing:
            print("\nMissing columns:", missing)
            return None

        return df

    except FileNotFoundError:
        print("\nERROR: social_media_performance.csv not found.")
        print("Keep the CSV file in the same folder as main.py.")

    except pd.errors.EmptyDataError:
        print("\nERROR: CSV file is empty.")

    except pd.errors.ParserError:
        print("\nERROR: CSV format is invalid.")

    except Exception as e:
        print("\nERROR:", e)

    return None


# ============================================================
# B. PYTHON CORE CONCEPTS
# ============================================================

def python_core(df):

    print("\n========== A. PYTHON CORE CONCEPTS ==========")

    project = "Social Media Analytics"
    total_posts = len(df)

    print("Project:", project)
    print("Total Posts:", total_posts)

    print("\nFirst 5 Posts:")

    for i in range(min(5, len(df))):
        print(
            df["post_id"].iloc[i],
            "->",
            df["platform"].iloc[i]
        )

    likes = int(df["likes"].iloc[0])
    comments = int(df["comments"].iloc[0])
    shares = int(df["shares"].iloc[0])

    total_engagement = likes + comments + shares

    print("\nFirst Post:")
    print("Likes:", likes)
    print("Comments:", comments)
    print("Shares:", shares)
    print("Total Engagement:", total_engagement)

    if total_posts > 0:
        print("Dataset contains social media posts.")


# ============================================================
# C. DATA STRUCTURES
# ============================================================

def data_structures(df):

    print("\n========== B. DATA STRUCTURES ==========")

    # List
    platforms = df["platform"].unique().tolist()

    print("\nList of Platforms:")
    print(platforms)

    # Tuple
    content_types = tuple(
        df["content_type"].unique()
    )

    print("\nTuple of Content Types:")
    print(content_types)

    # Set
    topics = set(
        df["topic"].dropna()
    )

    print("\nSet of Topics:")
    print(topics)

    # Dictionary
    post = df.iloc[0].to_dict()

    print("\nDictionary of First Post:")

    for key, value in post.items():
        print(key, ":", value)


# ============================================================
# D. PYTHON LIBRARIES
# ============================================================

def python_libraries(df):

    print("\n========== C. PYTHON LIBRARIES ==========")

    print("\nPandas:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    views = np.array(
        df["views"],
        dtype=float
    )

    print("\nNumPy:")
    print("Average Views:", round(np.mean(views), 2))
    print("Maximum Views:", np.max(views))
    print("Minimum Views:", np.min(views))

    print("\nLibraries used:")
    print("Pandas  - Data analysis")
    print("NumPy   - Numerical calculations")
    print("SciPy   - Statistical analysis")
    print("Requests - Web requests")
    print("BeautifulSoup - Web scraping")


# ============================================================
# E. CUSTOM MODULE
# ============================================================

def custom_module(df):

    print("\n========== D. CUSTOM MODULE ==========")

    row = df.iloc[0]

    likes = int(row["likes"])
    comments = int(row["comments"])
    shares = int(row["shares"])
    rate = float(row["engagement_rate"])

    total = analytics_module.calculate_engagement(
        likes,
        comments,
        shares
    )

    category = analytics_module.engagement_category(
        rate
    )

    print("\nPost ID:", row["post_id"])
    print("Platform:", row["platform"])
    print("Likes:", likes)
    print("Comments:", comments)
    print("Shares:", shares)
    print("Total Engagement:", total)
    print("Engagement Rate:", rate)
    print("Category:", category)


# ============================================================
# F. EXCEPTION HANDLING
# ============================================================

def exception_handling(df):

    print("\n========== E. EXCEPTION HANDLING ==========")

    try:

        platform = input(
            "\nEnter platform: "
        ).strip()

        result = df[
            df["platform"].str.lower()
            == platform.lower()
        ]

        if result.empty:
            print("Platform not found.")

        else:
            print(
                "Number of Posts:",
                len(result)
            )

            print(
                "Average Views:",
                round(result["views"].mean(), 2)
            )

            print(
                "Average Likes:",
                round(result["likes"].mean(), 2)
            )

    except KeyError as e:
        print("Column error:", e)

    except ValueError:
        print("Invalid value.")

    except Exception as e:
        print("Error:", e)


# ============================================================
# G. WEB SCRAPING
# ============================================================

def web_scraping():

    print("\n========== F. WEB SCRAPING ==========")

    try:

        import requests
        from bs4 import BeautifulSoup

        url = "https://en.wikipedia.org/wiki/Social_media"

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        print("\nSocial Media Web Page:")

        print(
            soup.title.get_text(
                strip=True
            )
        )

        print("\nFirst 5 Headings:")

        count = 0

        for heading in soup.find_all(
            ["h2", "h3"]
        ):

            text = heading.get_text(
                strip=True
            )

            if text:
                print("-", text)
                count += 1

                if count == 5:
                    break

    except ImportError:

        print(
            "\nInstall required packages:"
        )

        print(
            "python -m pip install requests beautifulsoup4"
        )

    except Exception as e:

        print(
            "\nWeb scraping error:",
            e
        )


# ============================================================
# H. FILE HANDLING
# ============================================================

def file_handling(df):

    print("\n========== G. FILE HANDLING ==========")

    try:

        with open(
            FILE,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(file)

            header = next(reader)
            first_row = next(reader)

        print("\nCSV Headers:")
        print(header)

        print("\nFirst Record:")
        print(first_row)

        backup = "social_media_backup.csv"

        df.head(10).to_csv(
            backup,
            index=False
        )

        print(
            "\nBackup created:",
            backup
        )

    except FileNotFoundError:
        print("File not found.")

    except StopIteration:
        print("CSV contains no records.")

    except Exception as e:
        print("File error:", e)


# ============================================================
# I. CRUD OPERATIONS
# ============================================================

def crud_operations(df):

    while True:

        print("\n========== H. CRUD OPERATIONS ==========")

        print("1. Create Post")
        print("2. Read Post")
        print("3. Update Post")
        print("4. Delete Post")
        print("5. Back")

        choice = input(
            "\nEnter choice: "
        ).strip()

        # CREATE
        if choice == "1":

            try:

                print("\nEnter New Post")

                post_id = input("Post ID: ").strip()
                platform = input("Platform: ").strip()
                content_type = input("Content Type: ").strip()
                topic = input("Topic: ").strip()

                language = input(
                    "Language [EN]: "
                ).strip() or "EN"

                region = input(
                    "Region [IN]: "
                ).strip() or "IN"

                post_datetime = input(
                    "Post Date/Time: "
                ).strip()

                hashtags = input(
                    "Hashtags: "
                ).strip()

                sentiment = float(
                    input("Sentiment Score: ")
                )

                views = int(
                    input("Views: ")
                )

                likes = int(
                    input("Likes: ")
                )

                comments = int(
                    input("Comments: ")
                )

                shares = int(
                    input("Shares: ")
                )

                viral = int(
                    input(
                        "Viral? 1=Yes, 0=No: "
                    )
                )

                if views > 0:

                    engagement_rate = (
                        likes
                        + comments
                        + shares
                    ) / views

                else:
                    engagement_rate = 0

                new_post = {
                    "post_id": post_id,
                    "platform": platform,
                    "content_type": content_type,
                    "topic": topic,
                    "language": language,
                    "region": region,
                    "post_datetime": post_datetime,
                    "hashtags": hashtags,
                    "sentiment_score": sentiment,
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "engagement_rate": engagement_rate,
                    "is_viral": viral
                }

                df.loc[len(df)] = new_post

                df.to_csv(
                    FILE,
                    index=False
                )

                print(
                    "\nPost created successfully."
                )

            except ValueError:
                print(
                    "\nEnter valid numeric values."
                )

            except Exception as e:
                print("Create error:", e)

        # READ
        elif choice == "2":

            post_id = input(
                "\nEnter Post ID: "
            ).strip()

            result = df[
                df["post_id"].astype(str)
                == post_id
            ]

            if result.empty:
                print("Post not found.")

            else:
                print(
                    result.to_string(
                        index=False
                    )
                )

        # UPDATE
        elif choice == "3":

            post_id = input(
                "\nEnter Post ID: "
            ).strip()

            indexes = df.index[
                df["post_id"].astype(str)
                == post_id
            ].tolist()

            if not indexes:

                print("Post not found.")

            else:

                try:

                    index = indexes[0]

                    likes = int(
                        input(
                            "New Likes: "
                        )
                    )

                    comments = int(
                        input(
                            "New Comments: "
                        )
                    )

                    shares = int(
                        input(
                            "New Shares: "
                        )
                    )

                    df.at[
                        index,
                        "likes"
                    ] = likes

                    df.at[
                        index,
                        "comments"
                    ] = comments

                    df.at[
                        index,
                        "shares"
                    ] = shares

                    views = float(
                        df.at[
                            index,
                            "views"
                        ]
                    )

                    if views > 0:

                        df.at[
                            index,
                            "engagement_rate"
                        ] = (
                            likes
                            + comments
                            + shares
                        ) / views

                    df.to_csv(
                        FILE,
                        index=False
                    )

                    print(
                        "\nPost updated successfully."
                    )

                except ValueError:
                    print(
                        "Enter valid numbers."
                    )

        # DELETE
        elif choice == "4":

            post_id = input(
                "\nEnter Post ID: "
            ).strip()

            indexes = df.index[
                df["post_id"].astype(str)
                == post_id
            ].tolist()

            if not indexes:

                print("Post not found.")

            else:

                df.drop(
                    indexes[0],
                    inplace=True
                )

                df.reset_index(
                    drop=True,
                    inplace=True
                )

                df.to_csv(
                    FILE,
                    index=False
                )

                print(
                    "\nPost deleted successfully."
                )

        elif choice == "5":
            break

        else:
            print("Invalid choice.")


# ============================================================
# J. NUMPY
# ============================================================

def numpy_analysis(df):

    print("\n========== I. NUMPY ANALYSIS ==========")

    views = np.array(
        df["views"],
        dtype=float
    )

    likes = np.array(
        df["likes"],
        dtype=float
    )

    comments = np.array(
        df["comments"],
        dtype=float
    )

    shares = np.array(
        df["shares"],
        dtype=float
    )

    engagement = (
        likes
        + comments
        + shares
    )

    print(
        "Average Views:",
        round(np.mean(views), 2)
    )

    print(
        "Maximum Views:",
        np.max(views)
    )

    print(
        "Minimum Views:",
        np.min(views)
    )

    print(
        "Average Likes:",
        round(np.mean(likes), 2)
    )

    print(
        "Average Comments:",
        round(np.mean(comments), 2)
    )

    print(
        "Average Shares:",
        round(np.mean(shares), 2)
    )

    print(
        "Average Total Engagement:",
        round(np.mean(engagement), 2)
    )


# ============================================================
# K. SCIPY
# ============================================================

def scipy_analysis(df):

    print("\n========== J. SCIPY ANALYSIS ==========")

    try:

        data = (
            df["engagement_rate"]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        result = stats.describe(data)

        print(
            "Number of Values:",
            result.nobs
        )

        print(
            "Minimum:",
            result.minmax[0]
        )

        print(
            "Maximum:",
            result.minmax[1]
        )

        print(
            "Mean:",
            result.mean
        )

        print(
            "Variance:",
            result.variance
        )

    except ImportError:

        print(
            "\nSciPy is not installed correctly."
        )

        print(
            "Run:"
        )

        print(
            "python -m pip install --upgrade scipy"
        )

    except Exception as e:
        print("SciPy error:", e)


# ============================================================
# L. PANDAS
# ============================================================
def pandas_analysis(df):
    print("First 5 rows:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nAverage Likes:", df["likes"].mean())
    print("Average Views:", df["views"].mean())

    print("\nPlatform Count:")
    print(df["platform"].value_counts())

# ============================================================
# M. DATA PREPROCESSING
# ============================================================

def data_preprocessing(df):

    print("\n========== L. DATA PREPROCESSING ==========")

    data = df.copy()

    print("\nOriginal Shape:", data.shape)

    print("\nMissing Values:")
    print(data.isnull().sum())

    duplicate_count = data.duplicated().sum()

    print(
        "\nDuplicate Rows:",
        duplicate_count
    )

    data = data.drop_duplicates()

    numeric_columns = [
        "views",
        "likes",
        "comments",
        "shares",
        "sentiment_score",
        "engagement_rate"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    for column in numeric_columns:

        data[column] = data[column].fillna(
            data[column].median()
        )

    categorical_columns = [
        "platform",
        "content_type",
        "topic",
        "language",
        "region"
    ]

    for column in categorical_columns:

        data[column] = data[column].fillna(
            "Unknown"
        )

    print(
        "\nProcessed Shape:",
        data.shape
    )

    print("\nPreprocessing completed.")
    print("\nData is ready for analysis and ML.")

    return data


# ============================================================
# N. DATA VISUALIZATION
# ============================================================

def data_visualization(df):

    print("\n========== M. DATA VISUALIZATION ==========")

    platform_counts = (
        df["platform"]
        .value_counts()
    )

    plt.figure()

    platform_counts.plot(
        kind="bar"
    )

    plt.title(
        "Number of Posts by Platform"
    )

    plt.xlabel("Platform")
    plt.ylabel("Number of Posts")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    avg_views = (
        df.groupby("platform")["views"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()

    avg_views.plot(
        kind="bar"
    )

    plt.title(
        "Average Views by Platform"
    )

    plt.xlabel("Platform")
    plt.ylabel("Average Views")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    plt.figure()

    df["sentiment_score"].plot(
        kind="hist",
        bins=20
    )

    plt.title(
        "Sentiment Score Distribution"
    )

    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    print(
        "\nThree visualizations displayed."
    )


# ============================================================
# O. STATISTICAL ANALYSIS
# ============================================================

def statistical_analysis(df):

    print("\n========== N. STATISTICAL ANALYSIS ==========")

    numeric_columns = [
        "views",
        "likes",
        "comments",
        "shares",
        "engagement_rate",
        "sentiment_score"
    ]

    print("\nDescriptive Statistics:")

    print(
        df[numeric_columns]
        .describe()
    )

    print(
        "\nCorrelation between Likes and Views:"
    )

    print(
        df[
            ["likes", "views"]
        ].corr()
    )

    print(
        "\nCorrelation between Shares and Views:"
    )

    print(
        df[
            ["shares", "views"]
        ].corr()
    )


# ============================================================
# P. PROBABILITY
# ============================================================

def probability_analysis(df):

    print("\n========== O. PROBABILITY ==========")

    total_posts = len(df)

    viral_posts = (
        df["is_viral"] == 1
    ).sum()

    if total_posts == 0:
        print("No posts available.")
        return

    probability_viral = (
        viral_posts / total_posts
    )

    print(
        "\nTotal Posts:",
        total_posts
    )

    print(
        "Viral Posts:",
        viral_posts
    )

    print(
        "Probability of a Random Post Being Viral:",
        round(probability_viral, 4)
    )

    platform = input(
        "\nEnter platform to calculate probability: "
    ).strip()

    platform_posts = (
        df["platform"].str.lower()
        == platform.lower()
    ).sum()

    if platform_posts > 0:

        probability = (
            platform_posts / total_posts
        )

        print(
            "Probability of Random Post Being",
            platform,
            ":",
            round(probability, 4)
        )

    else:
        print("Platform not found.")


# ============================================================
# Q. SAMPLING AND INFERENCE
# ============================================================

def sampling_inference(df):

    print("\n========== P. SAMPLING & INFERENCE ==========")

    sample_size = min(
        100,
        len(df)
    )

    if sample_size == 0:
        print("No data available.")
        return

    sample = df.sample(
        n=sample_size,
        random_state=42
    )

    print(
        "\nPopulation Size:",
        len(df)
    )

    print(
        "Sample Size:",
        len(sample)
    )

    population_mean = (
        df["likes"].mean()
    )

    sample_mean = (
        sample["likes"].mean()
    )

    sample_std = (
        sample["likes"].std()
    )

    if sample_size > 1:
        standard_error = (
            sample_std / np.sqrt(sample_size)
        )
    else:
        standard_error = 0

    confidence_interval = (
        1.96 * standard_error
    )

    lower = (
        sample_mean - confidence_interval
    )

    upper = (
        sample_mean + confidence_interval
    )

    print(
        "\nPopulation Mean Likes:",
        round(population_mean, 2)
    )

    print(
        "Sample Mean Likes:",
        round(sample_mean, 2)
    )

    print(
        "95% Confidence Interval:",
        "(",
        round(lower, 2),
        ",",
        round(upper, 2),
        ")"
    )


# ============================================================
# R. HYPOTHESIS TESTING
# ============================================================

def hypothesis_testing(df):

    print("\n========== Q. HYPOTHESIS TESTING ==========")

    viral = df[
        df["is_viral"] == 1
    ]["engagement_rate"].dropna()

    non_viral = df[
        df["is_viral"] == 0
    ]["engagement_rate"].dropna()

    if len(viral) == 0 or len(non_viral) == 0:

        print(
            "Both viral and non-viral groups are required."
        )

        return

    t_stat, p_value = stats.ttest_ind(
        viral,
        non_viral,
        equal_var=False
    )

    print(
        "\nH0: Mean engagement rate is equal"
    )

    print(
        "H1: Mean engagement rate is different"
    )

    print(
        "\nViral Mean:",
        round(viral.mean(), 6)
    )

    print(
        "Non-Viral Mean:",
        round(non_viral.mean(), 6)
    )

    print(
        "T-Statistic:",
        round(t_stat, 4)
    )

    print(
        "P-Value:",
        round(p_value, 6)
    )

    if p_value < 0.05:

        print(
            "\nResult: Reject H0 at 5% significance level."
        )

    else:

        print(
            "\nResult: Do not reject H0 at 5% significance level."
        )


# ============================================================
# S. MACHINE LEARNING
# ============================================================

def machine_learning(df):

    print("\n========== R. MACHINE LEARNING ==========")

    data = df.copy()

    features = [
        "platform",
        "content_type",
        "topic",
        "language",
        "region",
        "sentiment_score",
        "views",
        "likes",
        "comments",
        "shares"
    ]

    X = data[features]
    y = data["is_viral"]

    if y.nunique() < 2:
        print(
            "Machine Learning requires at least two classes in is_viral."
        )
        return

    categorical_features = [
        "platform",
        "content_type",
        "topic",
        "language",
        "region"
    ]

    numeric_features = [
        "sentiment_score",
        "views",
        "likes",
        "comments",
        "shares"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numeric_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            "\nTraining Records:",
            len(X_train)
        )

        print(
            "Testing Records:",
            len(X_test)
        )

        print(
            "\nAccuracy:",
            round(
                accuracy * 100,
                2
            ),
            "%"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_test,
                predictions
            )
        )

    except Exception as e:
        print("\nMachine Learning error:", e)


# ============================================================
# MAIN MENU
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    print("\n" + "=" * 60)
    print("           SOCIAL MEDIA ANALYTICS")
    print("=" * 60)

    print("\nDataset loaded successfully!")
    print("Records:", len(df))
    print("Columns:", len(df.columns))

    while True:

        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)

        print("1.  Python Core Concepts")
        print("2.  Data Structures")
        print("3.  Python Libraries")
        print("4.  Custom Module")
        print("5.  Exception Handling")
        print("6.  Web Scraping")
        print("7.  File Handling")
        print("8.  CRUD Operations")
        print("9.  NumPy")
        print("10. SciPy")
        print("11. Pandas")
        print("12. Data Preprocessing")
        print("13. Data Visualization")
        print("14. Statistical Analysis")
        print("15. Probability")
        print("16. Sampling & Inference")
        print("17. Hypothesis Testing")
        print("18. Machine Learning")
        print("19. Exit")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":
            python_core(df)

        elif choice == "2":
            data_structures(df)

        elif choice == "3":
            python_libraries(df)

        elif choice == "4":
            custom_module(df)

        elif choice == "5":
            exception_handling(df)

        elif choice == "6":
            web_scraping()

        elif choice == "7":
            file_handling(df)

        elif choice == "8":

            crud_operations(df)

            new_df = load_data()

            if new_df is not None:
                df = new_df

        elif choice == "9":
            numpy_analysis(df)

        elif choice == "10":
            scipy_analysis(df)

        elif choice == "11":
            pandas_analysis(df)

        elif choice == "12":

            df = data_preprocessing(df)

        elif choice == "13":
            data_visualization(df)

        elif choice == "14":
            statistical_analysis(df)

        elif choice == "15":
            probability_analysis(df)

        elif choice == "16":
            sampling_inference(df)

        elif choice == "17":
            hypothesis_testing(df)

        elif choice == "18":
            machine_learning(df)

        elif choice == "19":

            print(
                "\nProgram finished successfully."
            )

            break

        else:

            print(
                "\nInvalid choice."
            )

        input(
            "\nPress Enter to continue..."
        )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()