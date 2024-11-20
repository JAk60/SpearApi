# Configuration settings
loss_func_mask = [1, 1, 1, 1, 1, 1, 1]
batch_size = 32
lr_fm = 0.0005
lr_gm = 0.01
use_accuracy_score = True
feature_model = 'nn'
n_features = 768
n_hidden = 512
metric_avg = 'weighted'

# Parameters for fit_and_predict_proba
n_epochs = 100
start_len = 7
stop_len = 10
is_qt = True
is_qc = True
qt = 0.9
qc = 0.85