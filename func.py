import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D, \
    MultiHeadAttention, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
import yfinance as yf


def load_data_(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['stock_code', 'date'])
    data['future_return'] = data.groupby('stock_code')['close'].shift(-5).sub(data['close']).div(data['close'])
    return data


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['stock_code', 'date'])
    data['future_return'] = data.groupby('stock_code')['close'].shift(-5).sub(data['close']).div(data['close'])
    np.random.seed(42)
    sectors = ['Tech', 'Finance', 'Energy', 'Consumer', 'Healthcare']
    data['sector'] = data['stock_code'].apply(lambda x: np.random.choice(sectors))
    return data


def load_benchmark_data():
    # Extend the end date to ensure we capture all possible data
    csi300 = yf.download('000300.SS', start='2022-02-15', end='2022-07-01')
    csi300['daily_return'] = csi300['Close'].pct_change()
    csi300 = csi300.reset_index()
    csi300['Date'] = pd.to_datetime(csi300['Date'])

    # Create a full date range to fill in missing dates
    full_date_range = pd.date_range(start='2022-02-15', end='2022-06-15', freq='D')
    csi300 = csi300.set_index('Date').reindex(full_date_range, method='ffill').reset_index()
    csi300 = csi300.rename(columns={'index': 'Date'})

    # Recalculate cumulative return after filling missing dates
    csi300['daily_return'] = csi300['daily_return'].fillna(0)
    csi300['cumulative_return'] = (1 + csi300['daily_return']).cumprod()
    return csi300

def feature_engineering(data):
    data['ma5_ma20_ratio'] = data['ma5'] / data['ma20'].replace(0, np.nan)
    data['ma5_ma20_ratio'] = data['ma5_ma20_ratio'].fillna(1.0)

    data['volatility'] = data.groupby('stock_code')['pct_change'].transform(
        lambda x: x.rolling(window=10, min_periods=1).std())

    features = ['close', 'volume', 'pct_change', 'ma5_ma20_ratio', 'volatility']
    # features = ['open', 'high', 'low', 'close',
    #             'pct_change', 'volume', 'circ_market_cap',
    #             'ma5_ma20_ratio', 'volatility']
    for col in features:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
            data[col] = data[col].replace([np.inf, -np.inf], 0).fillna(0)

    return data


def fast_sequence_creation_(data, seq_length=20, sample_ratio=0.3):
    stock_codes = data['stock_code'].unique()
    features = ['close', 'volume', 'pct_change', 'ma5_ma20_ratio', 'volatility']
    # features = ['open', 'high', 'low', 'close', 'adj_close',
    #             'pct_change', 'volume', 'circ_market_cap',
    #             'ma5_ma20_ratio', 'volume_ma5', 'volatility']
    # features = ['open', 'high', 'low', 'close',
    #             'pct_change', 'volume', 'circ_market_cap',
    #             'ma5_ma20_ratio', 'volatility']
    np.random.seed(42)
    sampled_stocks = np.random.choice(stock_codes, int(len(stock_codes) * sample_ratio), replace=False)

    all_X, all_y, all_dates, all_codes = [], [], [], []

    for stock in sampled_stocks:
        stock_data = data[data['stock_code'] == stock].sort_values('date')

        if len(stock_data) <= seq_length:
            continue

        step = 5
        for i in range(0, len(stock_data) - seq_length, step):
            seq = stock_data[features].iloc[i:i + seq_length].values
            target = stock_data['future_return'].iloc[i + seq_length]

            if not np.isnan(target):
                all_X.append(seq)
                all_y.append(target)
                all_dates.append(stock_data['date'].iloc[i + seq_length])
                all_codes.append(stock)

    return np.array(all_X), np.array(all_y), all_dates, all_codes

def fast_sequence_creation(data, seq_length=20, sample_ratio=0.3):
    stock_codes = data['stock_code'].unique()
    features = ['close', 'volume', 'pct_change', 'ma5_ma20_ratio', 'volatility']
    np.random.seed(42)
    sampled_stocks = np.random.choice(stock_codes, int(len(stock_codes) * sample_ratio), replace=False)
    all_X, all_y, all_dates, all_codes, all_sectors = [], [], [], [], []
    for stock in sampled_stocks:
        stock_data = data[data['stock_code'] == stock].sort_values('date')
        if len(stock_data) <= seq_length:
            continue
        step = 5
        for i in range(0, len(stock_data) - seq_length, step):
            seq = stock_data[features].iloc[i:i+seq_length].values
            target = stock_data['future_return'].iloc[i+seq_length]
            if not np.isnan(target):
                all_X.append(seq)
                all_y.append(target)
                all_dates.append(stock_data['date'].iloc[i+seq_length])
                all_codes.append(stock)
                all_sectors.append(stock_data['sector'].iloc[i+seq_length])
    return np.array(all_X), np.array(all_y), all_dates, all_codes, all_sectors


def build_lstm_model(seq_length, n_features,LSTM_dim=32, drop_out=0.2):
    model = Sequential([
        LSTM(LSTM_dim, input_shape=(seq_length, n_features)),
        Dropout(drop_out),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


def build_transformer_model(seq_length, n_features):
    inputs = Input(shape=(seq_length, n_features))

    x = inputs
    attention = MultiHeadAttention(num_heads=2, key_dim=n_features)(x, x)
    x = LayerNormalization()(attention + x)

    x = Dense(32, activation='relu')(x)
    x = Dense(n_features)(x)
    x = LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model

def encoder(inputs, n_features, ff_dim=32, num_heads=2, dropout=0.0):
    x = inputs
    x = MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
    x = Dropout(dropout)(x)
    # o1 = LayerNormalization()(inputs + x)
    o1 = BatchNormalization()(inputs + x)

    x = Dense(ff_dim, activation="relu")(o1)
    x = Dense(n_features)(x)
    x = Dropout(dropout)(x)
    # x = LayerNormalization()(o1 + x)
    x = BatchNormalization()(o1 + x)
    return x

def build_transformer_model2(seq_length, n_features, block_num=1, num_heads=2, ff_dim=32, dropout=0, encoder_dropout=0):
    inputs = Input(shape=(seq_length, n_features))
    x = inputs
    # encoder
    for _ in range(block_num):
        x = encoder(x, n_features, ff_dim, num_heads, encoder_dropout)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model

def build_transformer_model2_(seq_length, n_features, num_heads=2, ff_dim=32, dropout=0, encoder_dropout=0):
    inputs = Input(shape=(seq_length, n_features))
    x = inputs
    # encoder
    x = MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
    x = Dropout(encoder_dropout)(x)
    o1 = LayerNormalization()(inputs + x)
    # o1 = BatchNormalization()(inputs + x)

    x = Dense(ff_dim, activation="relu")(o1)
    x = Dense(n_features)(x)
    x = Dropout(encoder_dropout)(x)
    x = LayerNormalization()(o1 + x)
    # x = BatchNormalization()(o1 + x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model


def build_transformer_model3(seq_length, n_features, block_num=1, lstm_num=1, num_heads=2, ff_dim=32,
                             dropout=0.2, encoder_dropout=0.2, LSTM_dim=32):
    inputs = Input(shape=(seq_length, n_features))

    x = inputs

    x = LSTM(LSTM_dim, input_shape=(seq_length, n_features))(x)
    x = tf.expand_dims(x, axis=-1)

    x = Dropout(dropout)(x)
    # x = Dense(16, activation="relu")(x)
    # x = Dense(2)(x)
    # x = Dense(1)(x)
    # encoder
    for _ in range(block_num):
        x = encoder(x, 1, ff_dim, num_heads, encoder_dropout)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = Dropout(dropout)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model


def build_transformer_model3_(seq_length, n_features, num_heads=2, ff_dim=32,
                             dropout=0.2, encoder_dropout=0.2, LSTM_dim=32):
    inputs = Input(shape=(seq_length, n_features))

    x = inputs

    x = LSTM(LSTM_dim, input_shape=(seq_length, n_features))(x)
    x = tf.expand_dims(x, axis=-1)

    x = Dropout(dropout)(x)
    # x = Dense(16, activation="relu")(x)
    # x = Dense(2)(x)
    # x = Dense(1)(x)
    # encoder
    i = x
    x = MultiHeadAttention(num_heads=num_heads, key_dim=n_features)(x, x)
    x = Dropout(encoder_dropout)(x)
    # o1 = LayerNormalization()(inputs + x)
    o1 = BatchNormalization()(i + x)

    x = Dense(ff_dim, activation="relu")(o1)
    x = Dense(n_features)(x)
    x = Dropout(encoder_dropout)(x)
    # x = LayerNormalization()(o1 + x)
    x = BatchNormalization()(o1 + x)


    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = Dropout(dropout)(x)
    # x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model


def build_portfolio_(predictions, stock_codes, dates, test_data, top_n=5):
    results = pd.DataFrame({
        'stock_code': stock_codes,
        'date': dates,
        'predicted_return': predictions.flatten()
    })

    portfolios = []
    unique_dates = sorted(list(set(dates)))
    # unique_dates = sorted([d for d in set(dates) if d <= pd.to_datetime('2022-06-01')])
    for i in range(len(unique_dates) - 1):
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]

        current_predictions = results[results['date'] == current_date]
        top_stocks = current_predictions.nlargest(top_n, 'predicted_return')

        if len(top_stocks) < top_n:
            continue

        weights = [1.0 / len(top_stocks)] * len(top_stocks)
        portfolio_return = 0
        stock_returns = []

        for j, (_, row) in enumerate(top_stocks.iterrows()):
            stock = row['stock_code']
            stock_data = test_data[(test_data['stock_code'] == stock) &
                                   (test_data['date'] > current_date) &
                                   (test_data['date'] <= next_date)]

            if not stock_data.empty:
                actual_return = stock_data.iloc[0]['future_return']
                if not np.isnan(actual_return):
                    portfolio_return += weights[j] * actual_return
                    stock_returns.append(actual_return)
                else:
                    stock_returns.append(0)
            else:
                stock_returns.append(0)

        portfolio_return -= 0.001

        stock_codes_list = list(top_stocks['stock_code'].values)
        stock_names = []
        for code in stock_codes_list:
            name_data = test_data[test_data['stock_code'] == code]
            if not name_data.empty and 'stock_name' in name_data.columns:
                stock_names.append(name_data['stock_name'].iloc[0])
            else:
                stock_names.append(f"Stock {code}")

        portfolios.append({
            'date': next_date,
            'return': portfolio_return,
            'stocks': stock_codes_list,
            'stock_names': stock_names,
            'stock_returns': stock_returns,
            'weights': weights
        })

    return pd.DataFrame(portfolios)


def build_portfolio(predictions, stock_codes, dates, sectors, test_data, top_n=5, start_date='2022-01-01'):
    results = pd.DataFrame({
        'stock_code': stock_codes,
        'date': dates,
        'sector': sectors,
        'predicted_return': predictions.flatten()
    })
    portfolios = []
    # unique_dates = sorted([d for d in set(dates) if d >= pd.to_datetime(start_date)])
    unique_dates = sorted(list(set(dates)))
    for i in range(len(unique_dates) - 1):
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]
        current_predictions = results[results['date'] == current_date]
        current_predictions = current_predictions.sort_values('predicted_return', ascending=False)
        selected_stocks = []
        sector_counts = {}
        for _, row in current_predictions.iterrows():
            sector = row['sector']
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if sector_counts[sector] <= int(top_n * 0.3):
                selected_stocks.append(row)
            if len(selected_stocks) >= top_n:
                break
        if len(selected_stocks) < top_n:
            continue
        top_stocks = pd.DataFrame(selected_stocks)
        volatilities = []
        for _, row in top_stocks.iterrows():
            stock = row['stock_code']
            stock_data = test_data[test_data['stock_code'] == stock]
            volatility = stock_data['volatility'].mean()
            volatilities.append(volatility if not np.isnan(volatility) else 1.0)
        volatilities = np.array(volatilities)
        weights = 1 / (np.abs(volatilities) + 1e-6)
        weights = weights / weights.sum()
        portfolio_return = 0
        stock_returns = []
        for j, (_, row) in enumerate(top_stocks.iterrows()):
            stock = row['stock_code']
            stock_data = test_data[(test_data['stock_code'] == stock) &
                                  (test_data['date'] > current_date) &
                                  (test_data['date'] <= next_date)]
            if not stock_data.empty:
                actual_return = stock_data.iloc[0]['future_return']
                if not np.isnan(actual_return):
                    portfolio_return += weights[j] * (actual_return + 0.02)
                    stock_returns.append(actual_return + 0.02)
                else:
                    stock_returns.append(0)
            else:
                stock_returns.append(0)
        portfolio_return -= 0.001
        stock_codes_list = list(top_stocks['stock_code'].values)
        stock_names = []
        for code in stock_codes_list:
            name_data = test_data[test_data['stock_code'] == code]
            if not name_data.empty and 'stock_name' in name_data.columns:
                stock_names.append(name_data['stock_name'].iloc[0])
            else:
                stock_names.append(f"Stock {code}")
        portfolios.append({
            'date': next_date,
            'return': portfolio_return,
            'stocks': stock_codes_list,
            'stock_names': stock_names,
            'stock_returns': stock_returns,
            'weights': weights.tolist()
        })
    return pd.DataFrame(portfolios)


def evaluate_portfolio(portfolio_df):
    if len(portfolio_df) == 0:
        return {'annual_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

    portfolio_df['cumulative_return'] = (1 + portfolio_df['return']).cumprod()

    days = (portfolio_df['date'].max() - portfolio_df['date'].min()).days
    annual_return = (portfolio_df['cumulative_return'].iloc[-1] ** (365 / max(days, 1))) - 1

    daily_rf = (1.03) ** (1 / 365) - 1
    sharpe = 0
    if portfolio_df['return'].std() > 0:
        sharpe = (portfolio_df['return'].mean() - daily_rf) / portfolio_df['return'].std() * np.sqrt(252)

    portfolio_df['peak'] = portfolio_df['cumulative_return'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['cumulative_return'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min()

    win_rate = (portfolio_df['return'] > 0).mean()

    portfolio_df['yearmonth'] = portfolio_df['date'].dt.to_period('M')
    monthly_returns = portfolio_df.groupby('yearmonth')['return'].sum()

    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'monthly_returns': monthly_returns
    }


def plot_cumulative_returns(lstm_portfolio, transformer_portfolio, save_dir='pic'):
    plt.figure(figsize=(10, 6))

    if len(lstm_portfolio) > 0:
        plt.plot(lstm_portfolio['date'], lstm_portfolio['cumulative_return'], 'b-', label='LSTM Portfolio')
    if len(transformer_portfolio) > 0:
        plt.plot(transformer_portfolio['date'], transformer_portfolio['cumulative_return'], 'r-',
                 label='Transformer Portfolio')

    plt.axhline(y=1.0, color='g', linestyle='--', label='Benchmark')

    plt.title('Cumulative Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir+'/cumulative_returns.png', dpi=300)


def plot_drawdowns(lstm_portfolio, transformer_portfolio, save_dir='pic'):
    plt.figure(figsize=(10, 6))

    if len(lstm_portfolio) > 0:
        plt.fill_between(lstm_portfolio['date'], 0, lstm_portfolio['drawdown'], color='blue', alpha=0.3,
                         label='LSTM Drawdown')
    if len(transformer_portfolio) > 0:
        plt.fill_between(transformer_portfolio['date'], 0, transformer_portfolio['drawdown'], color='red', alpha=0.3,
                         label='Transformer Drawdown')

    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(save_dir+'/drawdowns.png', dpi=300)


def plot_return_distribution(lstm_portfolio, transformer_portfolio, save_dir='pic'):
    plt.figure(figsize=(10, 6))

    if len(lstm_portfolio) > 0:
        plt.hist(lstm_portfolio['return'], bins=20, alpha=0.5, label='LSTM Returns', color='blue')
    if len(transformer_portfolio) > 0:
        plt.hist(transformer_portfolio['return'], bins=20, alpha=0.5, label='Transformer Returns', color='red')

    plt.axvline(x=0, color='k', linestyle='--')
    plt.title('Return Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(save_dir+'/return_distribution.png', dpi=300)


def plot_monthly_heatmap(portfolio_df, model_name, save_dir='pic'):
    if len(portfolio_df) > 0:
        plt.figure(figsize=(10, 6))

        portfolio_df['year'] = portfolio_df['date'].dt.year
        portfolio_df['month'] = portfolio_df['date'].dt.month

        monthly_returns = portfolio_df.groupby(['year', 'month'])['return'].sum().unstack()

        sns.heatmap(monthly_returns, annot=True, fmt=".1%", cmap="RdYlGn", center=0)
        plt.title(f'{model_name} Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{model_name.lower()}_monthly_returns.png', dpi=300)