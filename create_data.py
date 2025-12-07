import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
df['target'] = y

df.to_csv('data.csv', index=False)
print("data.csv created successfully")
