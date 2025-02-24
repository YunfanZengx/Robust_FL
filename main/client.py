import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

class Sass(torch.optim.Optimizer):
    def __init__(self, params,
                 init_step_size=1.0, theta=0.2,
                 gamma_decr=0.7, gamma_incr=1.25,
                 alpha_max=10.0, eps_f=0.0):
        defaults = dict(
            init_step_size=init_step_size,
            theta=theta,
            gamma_decr=gamma_decr,
            gamma_incr=gamma_incr,
            alpha_max=alpha_max,
            eps_f=eps_f
        )
        super(Sass, self).__init__(params, defaults)
        self.state["step_size"] = init_step_size

    def step(self, closure):
        if closure is None:
            raise RuntimeError("SASS requires a closure that returns the forward loss.")
        loss = closure()
        loss.backward()
        step_size = self.state["step_size"]
        grad_norm_sq = 0.0
        for group in self.param_groups:
            params_current = [p.data.clone() for p in group["params"]]
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm_sq += p.grad.data.norm()**2
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= step_size * p.grad.data
            loss_next = closure()
            lhs = loss_next
            rhs = loss - group["theta"] * step_size * grad_norm_sq + group["eps_f"]
            if lhs <= rhs:
                step_size = min(step_size * group["gamma_incr"], group["alpha_max"])
            else:
                step_size = step_size * group["gamma_decr"]
                for p, pcurr in zip(group["params"], params_current):
                    p.data.copy_(pcurr)
        self.state["step_size"] = step_size
        grad_norm = float(grad_norm_sq**0.5)
        self.state["grad_norm"] = grad_norm
        return loss

def evaluate_loss(model, dataset, device="cpu", batch_size=1000):
    """Compute average NLL test loss over 'dataset'."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
    avg_loss = total_loss / len(dataset)
    model.cpu()
    return avg_loss

def local_train_sass(model, train_dataset, device, optimizer, cid,
                     nn_passes_counters, train_loss_vs_passes, test_loss_vs_passes,
                     test_dataset, local_epochs=1, batch_size=32):
    """
    Train one client locally using SASS.
    """
    model.train()
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epoch_losses = []
    epoch_lrs = []
    all_grad_norms = []
    for _ in range(local_epochs):
        minibatch_losses = []
        minibatch_lrs = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            def closure():
                optimizer.zero_grad()
                output = model(data)
                return F.nll_loss(output, target)
            loss = optimizer.step(closure)
            minibatch_losses.append(loss.item())
            current_lr = optimizer.state["step_size"]
            minibatch_lrs.append(current_lr)
            current_gn = optimizer.state["grad_norm"]
            all_grad_norms.append(current_gn)
            nn_passes_counters[cid] += 1
        avg_loss = np.mean(minibatch_losses) if minibatch_losses else 0.0
        avg_lr = np.mean(minibatch_lrs) if minibatch_lrs else optimizer.state["step_size"]
        train_loss_vs_passes[cid].append((nn_passes_counters[cid], avg_loss))
        epoch_losses.append(avg_loss)
        epoch_lrs.append(avg_lr)
        test_l = evaluate_loss(model, test_dataset, device)
        test_loss_vs_passes[cid].append((nn_passes_counters[cid], test_l))
    model.cpu()
    avg_grad_norm = float(np.mean(all_grad_norms)) if all_grad_norms else 0.0
    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    mean_lr = float(np.mean(epoch_lrs)) if epoch_lrs else optimizer.state["step_size"]
    return model, [mean_loss], [mean_lr], avg_grad_norm

def local_train_sgd(model, train_dataset, device, optimizer, cid, nn_passes_counters,
                    train_loss_vs_passes, test_loss_vs_passes, test_dataset,
                    local_epochs=1, batch_size=32):
    """
    Local training using standard SGD.
    """
    model.train()
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epoch_losses = []
    epoch_lrs = []
    for _ in range(local_epochs):
        minibatch_losses = []
        minibatch_lrs = []
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            minibatch_losses.append(loss.item())
            current_lr = optimizer.param_groups[0]["lr"]
            minibatch_lrs.append(current_lr)
            nn_passes_counters[cid] += 1
        avg_loss = np.mean(minibatch_losses) if minibatch_losses else 0.0
        avg_lr = np.mean(minibatch_lrs) if minibatch_lrs else optimizer.param_groups[0]["lr"]
        train_loss_vs_passes[cid].append((nn_passes_counters[cid], avg_loss))
        epoch_losses.append(avg_loss)
        epoch_lrs.append(avg_lr)
        test_l = evaluate_loss(model, test_dataset, device)
        test_loss_vs_passes[cid].append((nn_passes_counters[cid], test_l))
    model.cpu()
    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    mean_lr = float(np.mean(epoch_lrs)) if epoch_lrs else optimizer.param_groups[0]["lr"]
    return model, [mean_loss], [mean_lr], 0.0  # grad norm not applicable
