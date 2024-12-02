
import sys

from data_finefoods import load_graph
from models.score import compute_evaluation_metrics
import csv
import time
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import precision_recall_curve, roc_curve

from torch.utils.tensorboard import SummaryWriter
import datetime

import torch

from models.data import BipartiteData
from models.net_sample import CL-BGADSampled
from models.sampler import BipartiteNeighborSampler
from models.sampler import EdgePredictionSampler
from models.loss import reconstruction_loss
from models.score import compute_anomaly_score, edge_prediction_metric

from utils.sum_dict import dict_addto, dict_div
from utils.seed import seed_all
import numpy as np
# %% args

parser = argparse.ArgumentParser(description="CL-BGAD")
parser.add_argument("--name", type=str, default="finefoods_anomaly", help="name")
parser.add_argument(
    "--key", type=str, default="graph_anomaly_list", help="key to the data"
)
parser.add_argument("--id", type=int, default=0, help="id to the data")
parser.add_argument("--batch-size", type=int, default=2048, help="batch size")
parser.add_argument(
    "--num-neighbors-u",
    type=int,
    default=10,
    help="number of neighbors for node u in sampling",
)
parser.add_argument(
    "--num-neighbors-v",
    type=int,
    default=10,
    help="number of neighbors for node v in sampling",
)
parser.add_argument("--n-epoch", type=int, default=50, help="number of epoch")
parser.add_argument(
    "--scheduler-milestones",
    nargs="+",
    type=int,
    default=[20, 35],
    help="scheduler milestone",
)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument(
    "--score-agg", type=str, default="max", help="aggregation for node anomaly score"
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="number of workers in neighborhood sampling loader",
)

args1 = vars(parser.parse_args())

args2 = {
    "hidden_channels": 32,
    "latent_channels_u": 32,
    "latent_channels_v": 32,
    "edge_pred_latent": 32,
    "n_layers_encoder": 2,
    "n_layers_decoder": 2,
    "n_layers_mlp": 2,
    "dropout_prob": 0.0,
    "gamma": 0.2,
    "xe_loss_weight": 1.0,
    "structure_loss_weight": 0.2,
    "structure_loss_weight_anomaly_score": 0.2,
    "iter_check": 10,
    "seed": 0,
    "neg_sampler_mult": 3,
    "k_check": 15,
    "tensorboard": False,
    "progress_bar": False,
}

args = {**args1, **args2}

seed_all(args["seed"])

# result_dir = "results2/"
result_dir = "autodl-tmp/results2/datasave"


# %% params
batch_size = args["batch_size"]

# %% data
data = load_graph(args["name"], args["key"], args["id"])
print(data)

u_ch = data.xu.shape[1]
v_ch = data.xv.shape[1]
e_ch = data.xe.shape[1]

print(
    f"Data dimension: U node = {data.xu.shape}; V node = {data.xv.shape}; E edge = {data.xe.shape}; \n"
)

# %% model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphBEANSampled(
    in_channels=(u_ch, v_ch, e_ch),
    hidden_channels=args["hidden_channels"],
    latent_channels=(args["latent_channels_u"], args["latent_channels_v"]),
    edge_pred_latent=args["edge_pred_latent"],
    n_layers_encoder=args["n_layers_encoder"],
    n_layers_decoder=args["n_layers_decoder"],
    n_layers_mlp=args["n_layers_mlp"],
    dropout_prob=args["dropout_prob"],
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args["scheduler_milestones"], gamma=args["gamma"]
)

xu, xv = data.xu, data.xv
xe, adj = data.xe, data.adj
yu, yv, ye = data.yu, data.yv, data.ye

# sampler
train_loader = BipartiteNeighborSampler(
    adj,
    n_layers=4,
    base="v",
    batch_size=batch_size,
    drop_last=True,
    n_other_node=-1,
    num_neighbors_u=args["num_neighbors_u"],
    num_neighbors_v=args["num_neighbors_v"],
    num_workers=args["num_workers"],
    shuffle=True,
)

print(args)
print()

# %% train
def train(epoch, check_counter):

    model.train()

    n_batch = len(train_loader)
    if args["progress_bar"]:
        pbar = tqdm(total=n_batch, leave=False)
        pbar.set_description(f"#{epoch:3d}")

    total_loss = 0
    total_epred_metric = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
    total_loss_component = {"xu": 0.0, "xv": 0.0, "xe": 0.0, "e": 0.0, "total": 0.0}
    num_update = 0

    for batch_size, indices, adjacencies, e_flags, neg_samples in train_loader:

        # print(f"# u nodes: {len(indices[0])} | # v nodes: {len(indices[1])} | # edges: {len(indices[2])}")

        adjacencies = [adj.to(device) for adj in adjacencies]
        e_flags = [fl.to(device) for fl in e_flags]
        u_id, v_id, e_id = indices

        # sample
        xu_sample = xu[u_id].to(device)
        xv_sample = xv[v_id].to(device)
        xe_sample = xe[e_id].to(device)

        # edge pred samples
        target_adj = adjacencies[-1].adj_e.adj
        edge_pred_sampler = EdgePredictionSampler(
            target_adj, mult=args["neg_sampler_mult"]
        )
        edge_pred_samples = edge_pred_sampler.sample().to(device)
        # 新增用于对比学习的负样本采样
        neg_samples = edge_pred_sampler.sample().to(device)

        # 前向传播
        optimizer.zero_grad()

        # start = time.time()
        out = model(
            xu=xu_sample,
            xv=xv_sample,
            xe=xe_sample,
            bean_adjs=adjacencies,
            e_flags=e_flags,
            edge_pred_samples=edge_pred_samples,
            neg_samples=neg_samples  # 传递负样本用于对比学习
        )
        # print(f"training : {time.time() - start} s")
        # 计算损失
        last_adj_e = adjacencies[-1].adj_e
        xu_target = xu[last_adj_e.u_id].to(device)
        xv_target = xv[last_adj_e.v_id].to(device)
        xe_target = xe[last_adj_e.e_id].to(device)

        loss, loss_component = reconstruction_loss(
            xu=xu_target,
            xv=xv_target,
            xe=xe_target,
            adj=last_adj_e.adj,
            edge_pred_samples=edge_pred_samples,
            out=out,
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
        )
        # 计算对比学习损失
        contrastive_loss = out.get("contrastive_loss", 0.0)

        # 更新总的损失
        total_loss += float(loss + contrastive_loss)

        loss.backward()
        optimizer.step()

        epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])

        total_loss += float(loss)
        total_epred_metric = dict_addto(total_epred_metric, epred_metric)

        # 更新 total_loss_component，保留原有的损失项，并增加对比学习损失
        total_loss_component = dict_addto(total_loss_component, loss_component)

        # 确保 total_loss_component 中存在 'contrastive' 键
        if "contrastive" not in total_loss_component:
            total_loss_component["contrastive"] = 0.0
        total_loss_component = dict_addto(total_loss_component, {"contrastive": contrastive_loss})

        num_update += 1

        if args["progress_bar"]:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": float(total_loss / num_update),
                    "contrastive_loss": float(contrastive_loss),
                    "ep acc": epred_metric["acc"],
                    "ep f1": epred_metric["f1"],
                }
            )

        if num_update == args["k_check"]:
            loss = total_loss / num_update
            loss_component = dict_div(total_loss_component, num_update)
            epred_metric = dict_div(total_epred_metric, num_update)

            # tensorboard
            if args["tensorboard"]:
                tb.add_scalar("loss", loss, check_counter)
                tb.add_scalar("loss_xu", loss_component["xu"], check_counter)
                tb.add_scalar("loss_xv", loss_component["xv"], check_counter)
                tb.add_scalar("loss_xe", loss_component["xe"], check_counter)
                tb.add_scalar("loss_e", loss_component["e"], check_counter)

                tb.add_scalar("epred_acc", epred_metric["acc"], check_counter)
                tb.add_scalar("epred_f1", epred_metric["f1"], check_counter)
                tb.add_scalar("epred_prec", epred_metric["prec"], check_counter)
                tb.add_scalar("epred_rec", epred_metric["rec"], check_counter)

            check_counter += 1

            total_loss = 0
            total_epred_metric = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
            total_loss_component = {
                "xu": 0.0,
                "xv": 0.0,
                "xe": 0.0,
                "e": 0.0,
                "total": 0.0,
            }
            num_update = 0

    if args["progress_bar"]:
        pbar.close()
    scheduler.step()

    return loss, loss_component, epred_metric, check_counter


# %% evaluate and store
def eval(epoch):
    model.eval()
    start = time.time()

    print(f"Evaluation for epoch {epoch}")

    try:
        edge_pred_sampler = EdgePredictionSampler(adj, mult=args["neg_sampler_mult"])
        edge_pred_samples = edge_pred_sampler.sample().to(device)

        with torch.no_grad():
            out = model.inference(
                xu, xv, xe, adj, edge_pred_samples,
                batch_sizes=(2 ** 13, 2 ** 13, 2 ** 13),
                device=device,
                progress_bar=args["progress_bar"],
            )

            true_labels = edge_pred_samples.storage.value().cpu().numpy()
            eprob = out["eprob"].cpu().numpy()

            print(f"Shape of true_labels: {true_labels.shape}")
            print(f"Shape of eprob: {eprob.shape}")
            print(f"Range of true_labels: [{true_labels.min()}, {true_labels.max()}]")
            print(f"Range of eprob: [{eprob.min()}, {eprob.max()}]")

            # 初始化变量
            loss = None
            loss_component = None
            epred_metric = None
            eval_metrics = {
                'u_roc_auc': 0, 'v_roc_auc': 0, 'e_roc_auc': 0,
                'u_pr_auc': 0, 'v_pr_auc': 0, 'e_pr_auc': 0
            }

            # 其他评估代码
            loss, loss_component = reconstruction_loss(
                xu, xv, xe, adj, edge_pred_samples, out,
                xe_loss_weight=args["xe_loss_weight"],
                structure_loss_weight=args["structure_loss_weight"],
            )

            epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])

            anomaly_score = compute_anomaly_score(
                xu, xv, xe, adj, edge_pred_samples, out,
                xe_loss_weight=args["xe_loss_weight"],
                structure_loss_weight=args["structure_loss_weight_anomaly_score"],
            )

            eval_metrics = compute_evaluation_metrics(
                anomaly_score, yu, yv, ye, agg=args["score_agg"]
            )

        elapsed = time.time() - start

        print(
            f"Eval, loss: {loss:.4f}, "
            + f"u auc-roc: {eval_metrics['u_roc_auc']:.4f}, v auc-roc: {eval_metrics['v_roc_auc']:.4f}, e auc-roc: {eval_metrics['e_roc_auc']:.4f}, "
            + f"u auc-pr {eval_metrics['u_pr_auc']:.4f}, v auc-pr {eval_metrics['v_pr_auc']:.4f}, e auc-pr {eval_metrics['e_pr_auc']:.4f} "
            + f"> {elapsed:.2f}s"
        )

        return loss, loss_component, epred_metric

    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


# %% run training
loss_hist = []
loss_component_hist = []
epred_metric_hist = []

# tensor board
if args["tensorboard"]:
    log_dir = (
        "/logs/tensorboard/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + args["name"]
    )
    tb = SummaryWriter(log_dir=log_dir, comment=args["name"])
check_counter = 0

# eval(0)

for epoch in range(args["n_epoch"]):

    start = time.time()
    loss, loss_component, epred_metric, check_counter = train(epoch, check_counter)
    elapsed = time.time() - start

    loss_hist.append(loss)
    loss_component_hist.append(loss_component)
    epred_metric_hist.append(epred_metric)

    print(
        f"#{epoch:3d}, "
        + f"Loss: {loss:.4f} => xu: {loss_component['xu']:.4f}, xv: {loss_component['xv']:.4f}, "
        + f"xe: {loss_component['xe']:.4f}, "
        + f"e: {loss_component['e']:.4f} -> "
        + f"[acc: {epred_metric['acc']:.3f}, f1: {epred_metric['f1']:.3f} -> "
        + f"prec: {epred_metric['prec']:.3f}, rec: {epred_metric['rec']:.3f}] "
        + f"> {elapsed:.2f}s"
    )

    # if epoch % args["iter_check"] == 0:  # and epoch != 0:
    #     # tb eval
    #     eval(epoch)
    # 仅在第50次epoch时进行eval
    if epoch == 50:
        eval(epoch)


# %% after training
res = eval(args["n_epoch"])
ev_loss, ev_loss_component, ev_epred_metric = res

if args["tensorboard"]:
    tb.add_hparams(
        args,
        {
            "loss": ev_loss,
            "xu": ev_loss_component["xu"],
            "xv": ev_loss_component["xv"],
            "xe": ev_loss_component["xe"],
            "e": ev_loss_component["e"],
            "acc": ev_epred_metric["acc"],
            "f1": ev_epred_metric["f1"],
            "prec": ev_epred_metric["prec"],
            "rec": ev_epred_metric["rec"],
        },
    )

print()
print(args)