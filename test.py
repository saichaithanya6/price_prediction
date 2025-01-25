import pandas as pd
from sklearn.decomposition import PCA

# Example dataset (stocks as columns, daily prices as rows)
stock_prices = pd.DataFrame({
    'Stock_A': [100, 102, 104, 105],
    'Stock_B': [50, 51, 53, 54],
    'Stock_C': [200, 198, 202, 205],
})

# Apply PCA
pca = PCA(n_components=2)
stock_pca = pca.fit_transform(stock_prices)

print("Explained Variance Ratios:", pca.explained_variance_ratio_)