
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("Covid Data.csv")
print(df.head())
print(df.shape, df.columns.tolist())
print(df.describe(include="all"))

df = df.drop_duplicates()
data = df.select_dtypes(include=np.number).copy()

special_cols = [
    'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD',
    'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
    'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU'
]

for col in special_cols:
    if col in data.columns:
        data[col] = data[col].replace([97, 98, 99], np.nan)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(axis=1, how="all")
data = data.fillna(data.median())
data = data.loc[:, data.nunique() > 1]

print("\nFeatures:", data.columns.tolist())

if min(data.shape) < 3:
    raise ValueError("Need at least 3 rows and 3 features.")

scaled = StandardScaler().fit_transform(data)
pca = PCA()
pca_data = pca.fit_transform(scaled)
variance = pca.explained_variance_ratio_
cumulative = np.cumsum(variance)
print("\nExplained variance (%):", variance * 100)
print("Components for 95% variance:",
      np.argmax(cumulative >= 0.95) + 1)
for n in [2, 3]:
    retained = cumulative[n - 1]
    print(f"\n{n}D retained: {retained * 100:.2f}%")
    print(f"{n}D lost: {(1 - retained) * 100:.2f}%")
v2 = cumulative[1] * 100
v3 = cumulative[2] * 100
print("\n--- 2D vs 3D Comparison ---")
print(f"2D retains {v2:.2f}% variance and loses {100-v2:.2f}%.")
print(f"3D retains {v3:.2f}% variance and loses {100-v3:.2f}%.")
print(f"3D preserves {v3-v2:.2f} percentage points more variance.")
print("2D is simpler to visualize; 3D preserves more information.")
plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("COVID-19 PCA - 2D")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=150)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], s=3)
ax.set(
    xlabel="PC1",
    ylabel="PC2",
    zlabel="PC3",
    title="COVID-19 PCA - 3D"
)
plt.tight_layout()
plt.savefig("pca_3d.png", dpi=150)
plt.show()

pd.DataFrame(
    pca_data[:, :3],
    columns=["PC1", "PC2", "PC3"]
).to_csv("covid_pca_results.csv", index=False)
print("\nGraphs saved: pca_2d.png, pca_3d.png")
print("Results saved: covid_pca_results.csv")