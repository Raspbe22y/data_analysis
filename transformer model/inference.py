from scaled.model.health_regression_model import TransformerEncoderWithMLP
from scaled.dataset.health_dataset import HealthDataset
import torch

train_dataset = HealthDataset(
        output_csv='data/output_data_v1.csv',
        input_csv='data/input_data',
    )
val_dataset = HealthDataset(
    output_csv='data/output_data_v1.csv',
    input_csv='data/input_data',
)

inputs, targets = train_dataset[0]
inputs = inputs.to('cuda').unsqueeze(0)
targets = targets.to('cuda').unsqueeze(0)

model = TransformerEncoderWithMLP(
    seq_len=300,
    in_dim=17,
    out_dim=6,
    d_model=768,
    nhead=8,
    num_layers=16,
    pooling="mean",).to('cuda')

model.load_state_dict(torch.load('exp_output/trainning_health_regression_model/model-7000.pth'))
model.eval()
with torch.no_grad():
    result = model(inputs)
print("Predicted:", result)
# name = train_dataset.input_list[0]
# patient_id, season = name.split(".")[0].split("_")
# print("Predicted for:", patient_id, season)
