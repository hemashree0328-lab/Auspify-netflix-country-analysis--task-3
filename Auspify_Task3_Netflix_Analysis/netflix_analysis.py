
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set up project folders
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load the Netflix dataset
df = pd.read_csv(BASE_DIR / "Dataset.csv")

print("=" * 55)
print("AUSPIFY - NETFLIX COUNTRY-WISE CONTENT ANALYSIS")
print("=" * 55)
print(f"\nDataset loaded successfully: {df.shape[0]} rows")

# 3. Clean column names and remove duplicate records
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# 4. Clean country information
df["country"] = df["country"].fillna("").astype(str).str.strip()

# Save the cleaned dataset
df.to_csv(OUTPUT_DIR / "cleaned_netflix_data.csv", index=False)

# 5. Separate multiple countries listed for each title
country_data = df[df["country"] != ""].copy()

country_data["country"] = country_data["country"].str.split(",")
country_data = country_data.explode("country")
country_data["country"] = country_data["country"].str.strip()

country_data = country_data[country_data["country"] != ""]

# Avoid counting the same title-country combination more than once
country_data = country_data.drop_duplicates(
    subset=["show_id", "country"]
)

# 6. Count content by country
country_counts = (
    country_data.groupby("country")["show_id"]
    .nunique()
    .reset_index(name="content_count")
    .sort_values("content_count", ascending=False)
)

country_counts.to_csv(
    OUTPUT_DIR / "country_content_counts.csv", index=False
)

# 7. Display the top 10 countries
top_10 = country_counts.head(10)

print("\nTOP 10 COUNTRIES BY NETFLIX CONTENT")
print("-" * 45)
print(top_10.to_string(index=False))

# 8. Analyze Movies and TV Shows by country
type_counts = (
    country_data.groupby(["country", "type"])["show_id"]
    .nunique()
    .unstack(fill_value=0)
)

type_counts["total"] = type_counts.sum(axis=1)
type_counts = type_counts.sort_values("total", ascending=False)

type_counts.to_csv(OUTPUT_DIR / "country_type_breakdown.csv")

# 9. Create a bar chart of the top 10 countries
plt.figure(figsize=(12, 7))
plt.barh(top_10["country"][::-1], top_10["content_count"][::-1])
plt.title("Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_10_countries.png", dpi=300)
plt.close()

# 10. Create a chart comparing Movies and TV Shows
top_countries = type_counts.head(10)
plot_columns = [
    col for col in ["Movie", "TV Show"]
    if col in top_countries.columns
]

top_countries[plot_columns].plot(
    kind="bar",
    figsize=(12, 7)
)
plt.title("Movies vs TV Shows in Top 10 Countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "movies_vs_tv_shows.png", dpi=300)
plt.close()

# 11. Generate business insights
total_titles = df["show_id"].nunique()
missing_country = (df["country"] == "").sum()

with open(OUTPUT_DIR / "business_insights.txt", "w", encoding="utf-8") as file:
    file.write("AUSPIFY - NETFLIX COUNTRY-WISE ANALYSIS\n\n")
    file.write(f"Total unique titles analyzed: {total_titles}\n")
    file.write(f"Titles with missing country information: {missing_country}\n")
    file.write(f"Countries identified: {len(country_counts)}\n\n")

    file.write("TOP 10 COUNTRIES:\n")
    for _, row in top_10.iterrows():
        file.write(
            f"{row['country']}: {row['content_count']} titles\n"
        )

    file.write("\nBUSINESS INSIGHTS:\n")
    if not country_counts.empty:
        leader = country_counts.iloc[0]
        file.write(
            f"1. {leader['country']} ranks first, with "
            f"{leader['content_count']} titles.\n"
        )
        file.write(
            "2. Country-level counts include titles associated "
            "with multiple countries.\n"
        )
        file.write(
            "3. Missing country information may limit geographic analysis.\n"
        )
        file.write(
            "4. The Movie vs TV Show chart helps compare content types "
            "across leading countries.\n"
        )

print("\nANALYSIS COMPLETED SUCCESSFULLY!")
print(f"Output files saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print("-", file_path.name)
