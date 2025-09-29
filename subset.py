import pandas as pd

# Load the dataset
df = pd.read_csv("wiki_movie_plots_deduped.csv")

n_samples = min(500, len(df))
subset = df.sample(n=n_samples, random_state=42)[["Title", "Plot"]]

# Save the subset to a new CSV
subset.to_csv("subset_movie_plots.csv", index=False)

print("Subset saved to 'subset_movie_plots.csv'")