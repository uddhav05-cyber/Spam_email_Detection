# Spam Email Detection

A deep learning spam classifier using LSTM neural network trained on a balanced email dataset.

## Results
- **Test Accuracy: XX%**
- Balanced dataset (equal spam/ham samples via downsampling)
- Early stopping to prevent overfitting

## Tech Stack
Python, TensorFlow, Keras, NLTK, NumPy, Pandas, Matplotlib

## How it Works
1. **Preprocessing** — removes punctuation, strips stop words, tokenises text
2. **Balancing** — downsamples ham emails to match spam count
3. **Model** — Embedding → LSTM → Dense → Sigmoid
4. **Training** — Adam optimizer, EarlyStopping + ReduceLROnPlateau callbacks

## Model Architecture
```
Embedding → LSTM(16) → Dense(32, relu) → Dense(1, sigmoid)
```

## Usage
```python
predict("Congratulations! You won a free iPhone!")  # → 'spam'
predict("Hey, are we still meeting at 5pm?")        # → 'ham'
```

## How to Run
1. Clone the repo
2. Open `Detecting_Spam_Emails_Using_Tensorflow.ipynb` in Colab or Jupyter
3. Run all cells (Runtime → Run all)
