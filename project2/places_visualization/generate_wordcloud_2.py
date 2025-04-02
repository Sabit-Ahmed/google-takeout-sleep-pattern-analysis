import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

label = 'last_year'
# Load your labeled cluster data
df = pd.read_csv(f"results/cluster_centroids_labeled_{label}.csv")

# Check the column
if "category" not in df.columns:
    raise ValueError("The CSV must contain a 'category' column.")

# Count categories
category_counts = df["category"].value_counts().to_dict()

# Generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="tab10"
).generate_from_frequencies(category_counts)

# Save and show
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/category_wordcloud.png")
plt.show()

print(f"✅ Word cloud saved to results/category_wordcloud_{label}.png")
