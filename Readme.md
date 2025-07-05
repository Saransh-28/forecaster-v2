# 📘 forecaster-v2

An experimental framework for building ML-based trading bots with multi-timeframe feature engineering.

## 🎯 Overview

This repository explores techniques to:
- Generate technical & pattern-based features across **1 h**, **2 hr**, and **4 hr** timeframes.
- Define **trading targets** based on ATR or percentage thresholds.
- Train **classification models** (MLP) to predict multi-class outcomes: up/down/no move.
- Prototype pipelines including feature engineering, classification, and evaluation.

---

## 🚀 Quick Start

1. **Clone the repo**
    ```bash
    git clone https://github.com/Saransh-28/forecaster-v2.git
    cd forecaster-v2
    ```

2. **Install dependencies**
    ```bash
    pip install uv
    uv pip install -r pyproject.toml
    ```

3. **Prepare your OHLCV data**
   Load CSV files (e.g., BTC, ADA) into pandas DataFrames or fetch using ccxt binance 

4. **Generate features & targets**
    ```python
    from data_preprocessing.training_helpers import create_all_features
    train_btc = create_all_features(
        df=df_train_btc,
        token="btc",
        target_type="pct",
        tp=2,
        sl=0.8,
        max_bars=60*6,
        atr_period=60*6
    )
    ```

5. **Optionally merge multiple tokens**
    ```python
    from data_preprocessing.training_helpers import create_merged_feature
    merged = create_merged_feature(
        main_feature_df=train_btc,
        complementry_data_df=train_ada,
        complementry_token_name="ada",
        main_token_name="btc"
    )
    ```

6. **Train the model**
    ```python
    from model import get_model
    model = get_model(
        input_shape=(merged["X_combined_1h_2h_4h"].shape[1],),
        num_classes=3
    )
    model.fit(
        merged["X_combined_1h_2h_4h"].values,
        merged["y_combined_1h_2h_4h"].values,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[...]
    )
    ```

7. **Evaluate performance**
    ```python
    from evaluation import print_token_report
    print_token_report(
        model,
        {"test": (X_test, y_test_int)},
        classes=(0, 1, 2)
    )
    ```

---

## 🧠 Best Practices

- **Scale data** (e.g., `StandardScaler`) before training.
- **Use PCA** to reduce dimensionality.
- Monitor **train vs. validation loss** to detect overfitting.
- Leverage **Dropout**, **L2 regularization**, and **EarlyStopping**.
- Use **time-series splits** to prevent lookahead bias.
- Handle class imbalance via **class weighting** or resampling.
- Merge tokens only when there is predictive value across assets.

---

## 🔄 Future Enhancements

- Hyperparameter tuning with `GridSearchCV`.
- Online learning or incremental updates.
- Backtesting engine to validate live performance.
- Shift to regression for predicting return magnitude.
- Extend `strategies.py` with more feature modules.

---

## 🧑‍💻 Contributing

This is a personal experiment repo—but feedback and PRs are welcome! Consider adding:
- New signal or strategy modules
- Model improvements or alternative architectures
- Backtesting support

