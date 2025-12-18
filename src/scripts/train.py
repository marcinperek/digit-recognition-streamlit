import tomllib
from datetime import datetime
from tqdm import tqdm

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from utils.ml import train, validate
from ml.DigitRecognizer import DigitRecognizer




def  load_config(config_path: str) -> dict:
    """Load configuration from a TOML file."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config



def main():
    load_config("config.toml")
    config = load_config("config.toml")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set device
    device = torch.device(config["train"]["device"])

    # model
    model = DigitRecognizer().to(device)

    # load data
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), 
                                            torch.tensor(y_train, dtype=torch.long).to(device)), 
                                            batch_size=config["train"]["batch_size"], 
                                            shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device), 
                                           torch.tensor(y_test, dtype=torch.long).to(device)), 
                                           batch_size=config["train"]["batch_size"], 
                                           shuffle=False)

    # optimizer
    optimizer = Adam(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # training loop
    patience = 3
    best_val_loss = float('inf')
    for epoch in tqdm(range(config["train"]["epochs"])):
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train(model, train_loader, optimizer, loss_fn, config)

        val_loss, val_accuracy = validate(model, test_loader, loss_fn, config)

        checkpoint = model.state_dict()

        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, lr: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'config': config, 'model_state_dict': checkpoint}, f"{config["train"]["checkpoint_path"]}best_{run_timestamp}.pth")


        scheduler.step(val_loss)




if __name__ == "__main__":
    main()