import torch

def collect_state_action_data(env, policy, num_episodes=100):
    buffer = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            sa = torch.tensor(list(state) + list(action), dtype=torch.float32)
            buffer.append(sa)
            state = next_state
    return torch.stack(buffer)
