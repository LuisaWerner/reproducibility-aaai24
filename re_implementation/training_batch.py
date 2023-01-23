import torch
import torch.nn.functional as F


def train(model, optimizer, device, criterion):
    """
    training loop - trains specified model by computing batches
    returns average epoch accuracy and loss
    parameters are updated with gradient descent
    @param model: a model specified in model.py
    @param optimizer: torch.optimizer object
    @param device: gpu or cpu
    @param criterion: defined loss function
    """
    model.train()
    batch = model.data
    batch.to(device)
    optimizer.zero_grad()
    out = F.softmax(model(batch.x, batch.edge_index, batch.relations, batch.edge_weight), dim=-1)[batch.train_mask]
    loss = criterion(out, batch.y[batch.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, criterion, device, evaluator, mask):
    """
    validation loop. No gradient updates
    returns accuracy per epoch and loss
    @param model: a model specified in model.py
    @param evaluator: a evaluator instance
    @param mask: a mask with train/valid or test ids
    @param device: gpu or cpu
    @param criterion: defined loss function
    """

    model.eval()
    batch = model.data
    batch = batch.to(device)
    out = F.softmax(model(batch.x, batch.edge_index, batch.relations, batch.edge_weight), dim=-1)[mask]
    loss = criterion(out, model.data.y[mask])
    acc = evaluator.eval_acc_kenn(y_true=batch.y[mask], y_pred=out.argmax(dim=-1, keepdim=True)).cpu()

    return acc, loss
