from tqdm import tqdm
from torchvision import transforms
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Union
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import Tuple


def train_epoch(
    model: Module,
    data_loader: DataLoader,
    opt: Optimizer,
    criterion: _Loss,
    disable_pbar: bool = False,
) -> Tuple[float, float]:
    """
    Train for 1 epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    running_loss = correct = 0.0
    n_batches = len(data_loader)
    for (x, y) in tqdm(data_loader, ncols=80, disable=disable_pbar, leave=False):
        # if y.shape[0] < 128:
        #    continue

        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        pred_class = torch.argmax(pred, dim=-1)
        if y.ndim == 2:
            y = torch.argmax(y, dim=-1)
        correct += (pred_class == y).sum().item()

    loss = running_loss / n_batches
    acc = correct / len(data_loader.dataset)
    return loss, acc


def my_test(model, data_loader):
    """
    test accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
        acc = correct / len(data_loader.dataset)
    return acc



class KLDivLoss_custom(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super(KLDivLoss_custom, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return F.kl_div(
            F.log_softmax(input, dim=-1),
            target,
            reduction=self.reduction,
            log_target=self.log_target,
        )


data_transforms = transforms.Compose([transforms.RandomResizedCrop(256),])


def perturb(x_batch, bounds=[-1, 1]):
    x_batch = (x_batch - bounds[0]) / (bounds[1] - bounds[0])
    x_batch = x_batch.cpu()

    if x_batch.ndim == 3:
        x_batch = x_batch.unsqueeze(0)

    if x_batch.shape[1] == 1:
        normalize = transforms.Normalize([0.5], [0.5])
    else:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                15, translate=(0.1, 0.1), scale=(0.9, 1.0), shear=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    """
    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                25, translate=(0.2, 0.2), scale=(0.8, 1.0), shear=(0.1, 0.1)
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    """
    x_batch_mod = torch.stack([data_transforms(xi) for xi in x_batch], axis=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch_mod = x_batch_mod.to(device)

    return x_batch_mod


def adaptive_pred(model, x, n=5, bounds=[-1, 1], mode="normal"):
    ys = []
    diff_list = []
    y_orig = 0
    hash_list = []
    for i in range(n):
        x_mod = perturb(x, bounds)
        hash_mod = np.array(model.get_hash_list(x_mod))
        hash_list.append(hash_mod)
        if mode == "normal":
            y = model(x_mod)
        elif mode == "normal_sim":
            y = model(x_mod, index=i)
            y_orig = model(x, index=i)
            y_orig = F.softmax(y_orig, dim=-1)
        elif mode == "ideal_attack":
            y = model(x, index=i)
        elif mode == "ideal_defense":
            y = model(x_mod, x_hash=x)
        else:
            raise ValueError(f"invalid mode {mode}")

        y = F.softmax(y, dim=-1)

        diff = y_orig - y
        diff_abs_sum = torch.abs(diff).sum(dim=-1)
        diff_list.append(diff_abs_sum.cpu().numpy())
        ys.append(y)
    hash_np = np.stack(hash_list, axis=-1)
    num_unique = [len(np.unique(hi)) for hi in hash_np]
    num_unique_avg = np.mean(num_unique)
    diff_mean = np.mean(np.stack(diff_list))
    ys = torch.stack(ys, dim=0)
    return torch.mean(ys, dim=0), num_unique_avg


def knockoff_(
    T,
    S: Module,
    dataloader_sur: DataLoader,
    dataloader_test: DataLoader,
    opt: Optimizer,
    # sch: Optional[_LRScheduler] = None,
    acc_T: float = 1.0,
    batch_size: int = 128,
    epochs=20,
    disable_pbar=False,
    budget=50000,
    pred_type="soft",
    adaptive_mode="none",
    n_adaptive_queries=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # S = S.to(device)
    results = {"epochs": [], "accuracy": [], "accuracy_x": []}

    print("== Constructing Surrogate Dataset ==")
    xs = torch.tensor([])
    ys = torch.tensor([])
    T.eval()
    queries = 0
    unique_list = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader_sur, ncols=100, leave=True, disable=disable_pbar):
            x = x.to(device)
            if adaptive_mode != "none":
                if pred_type == "hard":
                    raise ValueError(
                        "adaptive attacks is only supported for pred_type=soft"
                    )
                y, n_unique = adaptive_pred(
                    T, x, mode=adaptive_mode, n=n_adaptive_queries
                )
                unique_list.append(n_unique)
            else:
                y = T(x)
                if pred_type == "soft":
                    y = F.softmax(y, dim=-1)
                else:
                    y = torch.argmax(y, dim=-1)

            xs = torch.cat((xs, x.cpu()), dim=0)
            ys = torch.cat((ys, y.cpu()), dim=0)
            queries += x.shape[0]
            if queries >= budget:
                break

    if pred_type == "hard":
        ys = ys.long()

    ds_knockoff = TensorDataset(xs, ys)

    dataloader_knockoff = torch.utils.data.DataLoader(
        ds_knockoff, batch_size=batch_size, num_workers=4, shuffle=True
    )

    print("\n== Training Clone Model ==")
    if pred_type == "soft":
        criterion = KLDivLoss_custom(reduction="batchmean")
    else:
        criterion = CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-9)
    for epoch in range(1, epochs + 1):

        loss_train, acc_train = train_epoch(
            S, dataloader_knockoff, opt, criterion, disable_pbar
        )
        acc_test = my_test(S, dataloader_test)
        print(
            "Epoch: {} Loss: {:.4f} Train Acc: {:.2f} Test Acc: {:.2f} ({:.2f}x) Loss: {:.4f}\n".format(
                epoch, loss_train, 100 * acc_train, 100 * acc_test, acc_test / acc_T,opt.state_dict()['param_groups'][0]['lr']
            )
        )
        if scheduler:
            scheduler.step()
        results["epochs"].append(epoch)
        results["accuracy"].append(acc_test)
        results["accuracy_x"].append(acc_test / acc_T)

    """
    exp_path = f"./exp/{args.dataset_tar}/{args.exp_id}/"
    df = pd.DataFrame(data=results)
    savedir_csv = exp_path + "csv/"
    if not os.path.exists(savedir_csv):
        os.makedirs(savedir_csv)
    df.to_csv(savedir_csv + "/knockoffnets.csv")
    """
    return
