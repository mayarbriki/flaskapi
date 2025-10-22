import matplotlib.pyplot as plt
import numpy as np

# Placeholder: Simulated form points for two teams over 10 matchweeks
matchweeks = np.arange(1, 11)
home_form = np.random.randint(0, 3, size=10).cumsum()
away_form = np.random.randint(0, 3, size=10).cumsum()

# 1. Form over time chart
plt.figure(figsize=(8, 4))
plt.plot(matchweeks, home_form, marker='o', label='Home Team Form')
plt.plot(matchweeks, away_form, marker='o', label='Away Team Form')
plt.title('Team Form Over Time')
plt.xlabel('Matchweek')
plt.ylabel('Form Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Prediction probabilities bar chart (placeholder values)
probs = [0.62, 0.23, 0.15]  # H / D / A
labels = ['Home Win', 'Draw', 'Away Win']
plt.figure(figsize=(6, 4))
plt.bar(labels, probs, color=['green', 'gray', 'red'])
plt.title('Match Outcome Prediction Probabilities')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()

# 3. Heat summary (placeholder: random matrix)
heat_data = np.random.rand(5, 5)
plt.figure(figsize=(5, 4))
plt.imshow(heat_data, cmap='hot', interpolation='nearest')
plt.title('Feature Heat Summary')
plt.colorbar(label='Value')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.tight_layout()
plt.show()