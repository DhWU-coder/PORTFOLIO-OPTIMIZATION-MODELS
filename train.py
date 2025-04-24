from func import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# start training
file_path = "data_cleaned.csv"

data = load_data(file_path)
print(f"Data loaded: {len(data)} rows")

data = feature_engineering(data)
print("Feature engineering completed")

data_train = data[data['date'] < pd.to_datetime('2022-01-01')]
data_test = data[data['date'] >= pd.to_datetime('2022-02-01')]
print(f"Data split: {len(data_train)} train, {len(data_test)} test rows")

X_train, y_train, _, _ = fast_sequence_creation(data_train, sample_ratio=0.3)
X_test, y_test, test_dates, test_codes = fast_sequence_creation(data_test, sample_ratio=0.5)
print(f"Sequence data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

lstm_model = build_lstm_model(X_train.shape[1], X_train.shape[2])
print("Training LSTM model...")
lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2)],
    verbose=1
)
# save model
lstm_model.save('model/lstm/lstm.h5')

transformer_model = build_transformer_model2(seq_length=X_train.shape[1],
                                             n_features=X_train.shape[2],
                                             block_num=2,  # 2
                                             num_heads=16,  # 16
                                             ff_dim=64,  # 64
                                             dropout=0.2,  # 0.2
                                             encoder_dropout=0.1)
print("Training Transformer model...")
transformer_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=2)],
    verbose=1
)
# save model
transformer_model.save('model/transformer/transformer.h5')

lstm_transformer_model = build_transformer_model3_(seq_length=X_train.shape[1],
                                                   n_features=X_train.shape[2],
                                                   num_heads=8,  # 8
                                                   ff_dim=64,  # 64
                                                   dropout=0.2,  # 0.2
                                                   encoder_dropout=0.2,  # 0.2
                                                   LSTM_dim=32)  # 32
print("Training LSTM_Transformer model...")
lstm_transformer_model.fit(
    X_train, y_train,
    epochs=7,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)
# save model
lstm_transformer_model.save('model/lstm_transformer/lstm_transformer.h5')

