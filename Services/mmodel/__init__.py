import torch

model_path = "D:/Master/SS_2024_Thesis_ISA/Thesis/Work-docs/RAMS-TSAD/Mononito/trained_models/smd/machine-1-2/LOF_1.pth"
model = torch.load(model_path)  # Use CPU to avoid GPU requirements for inspection
print('hi')
print(model)  # Print the model architecture
