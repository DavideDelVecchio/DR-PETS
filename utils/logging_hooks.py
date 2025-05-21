from torch.utils.tensorboard import SummaryWriter

def log_evaluation_metrics(writer: SummaryWriter, metrics: dict, step: int):
    for key, value in metrics.items():
        writer.add_scalar(f"Eval/{key}", value, step)

def log_planner_summary(writer: SummaryWriter, episode: int, score: float, length: int):
    writer.add_scalar("Planner/EpisodeReward", score, episode)
    writer.add_scalar("Planner/EpisodeLength", length, episode)