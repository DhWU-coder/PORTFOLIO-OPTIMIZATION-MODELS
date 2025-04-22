from func_ import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# start training
file_path = "data_cleaned.csv"

data = load_data(file_path)
print(f"Data loaded: {len(data)} rows")

benchmark_df = load_benchmark_data()

data = feature_engineering(data)
print("Feature engineering completed")

data_test = data[data['date'] >= pd.to_datetime('2022-01-21')]
print(f"Data split: {len(data_test)} test rows")

X_test, y_test, test_dates, test_codes, test_sectors = fast_sequence_creation(data_test, sample_ratio=0.5)
print(f"Sequence data: {X_test.shape[0]} test samples")

print("Loading LSTM model...")
lstm_model = tf.keras.models.load_model('model/lstm/lstm27.h5')  # load model

print("Loading Transformer model...")
transformer_model = tf.keras.models.load_model('model/transformer/transformer32.h5')  # load model

print("Loading LSTM Transformer model...")
lstm_transformer_model = tf.keras.models.load_model('model/lstm_transformer/lstm_transformer33.h5')  # load model

print("Building portfolios...")
lstm_predictions = lstm_model.predict(X_test, verbose=0)
transformer_predictions = transformer_model.predict(X_test, verbose=0)
lstm_transformer_predictions = lstm_transformer_model.predict(X_test, verbose=0)

lstm_portfolio = build_portfolio_(lstm_predictions, test_codes, test_dates, data_test)
transformer_portfolio = build_portfolio_(transformer_predictions, test_codes, test_dates, data_test)
lstm_transformer_portfolio = build_portfolio_(lstm_transformer_predictions, test_codes, test_dates, data_test)

lstm_metrics = evaluate_portfolio(lstm_portfolio, benchmark_df)
transformer_metrics = evaluate_portfolio(transformer_portfolio, benchmark_df)
lstm_transformer_metrics = evaluate_portfolio(lstm_transformer_portfolio, benchmark_df)

print("\n--- LSTM Portfolio Performance ---")
if len(lstm_portfolio) > 0:
    print(f"Annual Return: {lstm_metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {lstm_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {lstm_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {lstm_metrics['win_rate']:.2%}")
    print(f"Excess Return (vs CSI 300): {lstm_metrics['excess_return']:.2%}")
    print(f"Information Ratio: {lstm_metrics['information_ratio']:.2f}")

    print("\nLSTM Selected Stocks (First 5 dates):")
    for i in range(min(5, len(lstm_portfolio))):
        date = lstm_portfolio['date'].iloc[i]
        stocks = lstm_portfolio['stock_names'].iloc[i]
        returns = lstm_portfolio['return'].iloc[i]
        print(f"Date: {date.strftime('%Y-%m-%d')}, Return: {returns:.2%}")
        for j, (stock, weight) in enumerate(zip(stocks, lstm_portfolio['weights'].iloc[i])):
            print(f"  {stock}: {weight:.2%}")

print("\n--- Transformer Portfolio Performance ---")
if len(transformer_portfolio) > 0:
    print(f"Annual Return: {transformer_metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {transformer_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {transformer_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {transformer_metrics['win_rate']:.2%}")
    print(f"Excess Return (vs CSI 300): {transformer_metrics['excess_return']:.2%}")
    print(f"Information Ratio: {transformer_metrics['information_ratio']:.2f}")

    print("\nTransformer Selected Stocks (First 5 dates):")
    for i in range(min(5, len(transformer_portfolio))):
        date = transformer_portfolio['date'].iloc[i]
        stocks = transformer_portfolio['stock_names'].iloc[i]
        returns = transformer_portfolio['return'].iloc[i]
        print(f"Date: {date.strftime('%Y-%m-%d')}, Return: {returns:.2%}")
        for j, (stock, weight) in enumerate(zip(stocks, transformer_portfolio['weights'].iloc[i])):
            print(f"  {stock}: {weight:.2%}")

print("\n--- LSTM+Transformer Portfolio Performance ---")
if len(lstm_transformer_portfolio) > 0:
    print(f"Annual Return: {lstm_transformer_metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {lstm_transformer_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {lstm_transformer_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {lstm_transformer_metrics['win_rate']:.2%}")
    print(f"Excess Return (vs CSI 300): {lstm_transformer_metrics['excess_return']:.2%}")
    print(f"Information Ratio: {lstm_transformer_metrics['information_ratio']:.2f}")

    print("\nTransformer Selected Stocks (First 5 dates):")
    for i in range(min(5, len(lstm_transformer_portfolio))):
        date = lstm_transformer_portfolio['date'].iloc[i]
        stocks = lstm_transformer_portfolio['stock_names'].iloc[i]
        returns = lstm_transformer_portfolio['return'].iloc[i]
        print(f"Date: {date.strftime('%Y-%m-%d')}, Return: {returns:.2%}")
        for j, (stock, weight) in enumerate(zip(stocks, lstm_transformer_portfolio['weights'].iloc[i])):
            print(f"  {stock}: {weight:.2%}")

plot_cumulative_returns_(lstm_portfolio, transformer_portfolio, lstm_transformer_portfolio, benchmark_df, save_dir='new_weight_pic')
plot_drawdowns_(lstm_portfolio, transformer_portfolio, lstm_transformer_portfolio, save_dir='new_weight_pic')
plot_return_distribution_(lstm_portfolio, transformer_portfolio, lstm_transformer_portfolio, save_dir='new_weight_pic')
plot_monthly_heatmap(lstm_portfolio, "LSTM", save_dir='new_weight_pic')
plot_monthly_heatmap(transformer_portfolio, "Transformer", save_dir='new_weight_pic')
plot_monthly_heatmap(lstm_transformer_portfolio, "LSTM_Transformer", save_dir='new_weight_pic')

print("Analysis complete. All visualizations saved separately.")

