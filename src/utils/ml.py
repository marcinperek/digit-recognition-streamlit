import torch


def train(model, train_loader, optimizer, loss_fn, config):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(config["train"]["device"]), batch_y.to(config["train"]["device"])
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, test_loader, loss_fn, config):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(config["train"]["device"]), batch_y.to(config["train"]["device"])
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy