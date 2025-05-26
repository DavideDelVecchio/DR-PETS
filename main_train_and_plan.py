import os, time, csv, json, torch, gym, numpy as np
from pathlib          import Path
from datetime         import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from planners.cem_planner import CEMPlanner
from models.dynamics_ensemble import DynamicsEnsemble
from models.density_model     import StateActionDensityModel
from utils.config_loader      import load_config

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
cfg              = load_config("configs/config.yaml")
state_dim        = cfg["experiment"]["state_dim"]
action_dim       = cfg["experiment"]["action_dim"]
ensemble_size    = cfg["experiment"]["ensemble_size"]
epsilon          = cfg["experiment"]["epsilon"]

episodes               = 20
horizon                = 200
retrain_epochs         = 50
batch_size             = 128
initial_random_eps     = 5
checkpoint_dir         = Path("checkpoints")
csv_path               = Path("episode_metrics.csv")
eval_lengths           = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
from utils.device import get_device
device = get_device(args.device if "args" in locals() else "auto")

dyn=None
den=None
planner= None
checkpoint_dir.mkdir(exist_ok=True)
# ─────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────
def append_to_csv(rowdict, path=csv_path):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rowdict.keys())
        if new_file: writer.writeheader()
        writer.writerow(rowdict)

# ─────────────────────────────────────────
# Cloud upload  (choose backend via env)
#   • export CLOUD_BUCKET="s3://my-bucket"    for AWS S3
#   • export CLOUD_BUCKET="gs://my-bucket"    for GCS
# ─────────────────────────────────────────
""" bucket_url = os.getenv("CLOUD_BUCKET")   # optional
def upload_to_cloud(local_path: Path):
    if not bucket_url: return
    if bucket_url.startswith("s3://"):
        import boto3, botocore
        s3 = boto3.client("s3")
        bname, keyprefix = bucket_url[5:].split("/",1) if "/" in bucket_url[5:] else (bucket_url[5:], "")
        key = f"{keyprefix}/{local_path.name}" if keyprefix else local_path.name
        s3.upload_file(str(local_path), bname, key)
    elif bucket_url.startswith("gs://"):
        from google.cloud import storage
        client = storage.Client()
        bname, keyprefix = bucket_url[5:].split("/",1) if "/" in bucket_url[5:] else (bucket_url[5:], "")
        bucket = client.bucket(bname)
        blob   = bucket.blob(f"{keyprefix}/{local_path.name}" if keyprefix else local_path.name)
        blob.upload_from_filename(local_path)
    else:
        print(f"[WARN] Unhandled CLOUD_BUCKET schema for {bucket_url}") """

# ─────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────
oh = lambda a: torch.nn.functional.one_hot(torch.tensor(a), num_classes=action_dim).float()
def to_loader(states, actions, nxt): return DataLoader(
        TensorDataset(torch.cat(states), torch.cat(actions), torch.cat(nxt)),
        batch_size=batch_size, shuffle=True)

def train_dyn(states, actions, nxt):
    model, opt = DynamicsEnsemble(state_dim, action_dim, ensemble_size).to(device), None
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(retrain_epochs):
        for s,a,sn in to_loader(states,actions,nxt):
            s,a,sn = s.to(device),a.to(device),sn.to(device)
            loss = ((model(s,a)-sn)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_den(states, actions):
    model = StateActionDensityModel(state_dim, action_dim).to(device)
    loader = DataLoader(torch.cat([torch.cat(states),torch.cat(actions)],dim=1),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(retrain_epochs):
        for batch in loader:
            batch=batch.to(device)
            loss=-model.flow.log_prob(batch).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def evaluate_different_lengths(planner, lengths, episodes=3):
    res = {L:[] for L in lengths}
    for L in lengths:
        env_tmp = gym.make("CartPole-v1"); env_tmp.env.length=L
        for _ in range(episodes):
            obs,_=env_tmp.reset(); total=0; done=False
            while not done:
                act,_=planner.plan(torch.tensor(obs).float().unsqueeze(0).to(device))
                obs,r,term,trunc,_=env_tmp.step(int(act.argmax()))
                total+=r; done=term or trunc
            res[L].append(total)
    return {L:np.mean(v) for L,v in res.items()}

# ─────────────────────────────────────────
# Dataset buffers (D)
# ─────────────────────────────────────────
s_buf,a_buf,sn_buf=[],[],[]
env = gym.make("CartPole-v1")

# Random episodes to seed D
for _ in range(initial_random_eps):
    obs,_=env.reset()
    for _ in range(horizon):
        act = env.action_space.sample()
        nxt,_,term,trunc,_=env.step(act)
        s_buf .append(torch.tensor(obs).float().unsqueeze(0))
        a_buf .append(oh(act).unsqueeze(0))
        sn_buf.append(torch.tensor(nxt).float().unsqueeze(0))
        obs=nxt
        if term or trunc: break

# TensorBoard
writer = SummaryWriter(log_dir=f"runs/drpets_{int(time.time())}")

# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
global_step=0
for ep in range(1, episodes+1):
    print(f"\n🔄 EPISODE {ep}/{episodes}   D size={len(s_buf)}")

    dyn = train_dyn(s_buf,a_buf,sn_buf)
    den = train_den(s_buf,a_buf)

    planner = CEMPlanner(
        dynamics_model = dyn,
        density_model  = den,
        epsilon        = epsilon,
        action_dim     = action_dim,
        horizon        = 30,
        num_samples    = 500,
        num_elites     = 50,
        num_iters      = 5
    )

    obs,_=env.reset(); ep_reward=0; step_times=[]
    for t in range(horizon):
        st = time.time()
        act_tensor, scores = planner.plan(torch.tensor(obs).float().unsqueeze(0).to(device))
        step_times.append(time.time()-st)
        act = int(act_tensor.argmax())
        nxt,r,term,trunc,_ = env.step(act)

        s_buf .append(torch.tensor(obs).float().unsqueeze(0))
        a_buf .append(oh(act).unsqueeze(0))
        sn_buf.append(torch.tensor(nxt).float().unsqueeze(0))

        ep_reward += r; obs = nxt; global_step+=1
        if term or trunc: break
    row = {
    "episode": ep,
    "reward" : ep_reward,
    "score_pets"  : scores["score_pets"],
    "score_drpets": scores["score_drpets"],
    "plan_latency": np.mean(step_times)}
    append_to_csv(row)
    # TensorBoard metrics
    writer.add_scalar("train/reward", ep_reward, ep)
    writer.add_scalar("debug/score_pets"  , scores["score_pets"]  , ep)
    writer.add_scalar("debug/score_drpets", scores["score_drpets"], ep)
    writer.add_scalar("debug/plan_latency" , np.mean(step_times)   , ep)

# Multi-length eval
dyn.eval()
den.eval()
eval_res = evaluate_different_lengths(planner, eval_lengths, episodes=50)
for L,avg_r in eval_res.items():
    writer.add_scalar(f"eval/len_{L:.2f}", avg_r, ep)



# Save checkpoints and upload
dyn_path = checkpoint_dir / f"dynamics_ep{ep}.pt"
den_path = checkpoint_dir / f"density_ep{ep}.pt"
torch.save(dyn.state_dict(), dyn_path)
torch.save(den.state_dict(), den_path)
# upload_to_cloud(dyn_path); upload_to_cloud(den_path)
# upload_to_cloud(csv_path)

print(f"✅ ep_reward={ep_reward} | mean plan latency={np.mean(step_times):.4f}s")

writer.close()
print("\nTraining complete! View logs with:\n  tensorboard --logdir runs")
print("CSV saved to", csv_path)
