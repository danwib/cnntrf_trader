import torch
from src.inference.decision import select_tau_by_grid_from_loader, evaluate_loader_cost_aware

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build loaders the same way as in training/eval (val/test with shuffle=False)
from src.training.data_module import build_dataloaders
loaders = build_dataloaders(master_dir="artifacts/datasets/master", seq_len=64, stride=1, batch_size=512, num_workers=4)
val_loader, test_loader = loaders["val"], loaders["test"]

# load your best checkpoint/model
ckpt = torch.load("artifacts/runs/<run>/checkpoints/best.pt", map_location=device)
from src.models.cnn_patchtst import CNNPatchTST, CNNPatchTSTConfig
model = CNNPatchTST(CNNPatchTSTConfig(**ckpt["model_cfg"])).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 1) Choose tau on validation (sigma-grid; cost 3 bps; horizon 4 bars)
best_tau, val_best, grid = select_tau_by_grid_from_loader(
    model, val_loader, device, horizon=4, cost_bps=3.0, grid="sigma",
    mults=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5), min_trades=200
)
print("Chosen tau:", best_tau, "Val metrics:", val_best)

# 2) Evaluate on test with that tau
test_metrics = evaluate_loader_cost_aware(model, test_loader, device, tau=best_tau, horizon=4, cost_bps=3.0)
print("Test metrics:", test_metrics)
