import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(
    {
        "length": [1.5, 0.5, 1.2, 0.9, 3],
        "width": [0.7, 0.2, 0.15, 0.2, 1.1],
    },
    index=["pig", "rabbit", "duck", "chicken", "horse"],
)
plot = df.plot(title="DataFrame Plot")
plt.show()