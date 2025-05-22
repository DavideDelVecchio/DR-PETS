import torch
import gym

def collect_state_action_data(env, policy, num_episodes=100):
    buffer = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            action_onehot = [1.0 if i == action else 0.0 for i in range(env.action_space.n)]
            sa = torch.tensor(list(state) + action_onehot, dtype=torch.float32)

            buffer.append(sa)
            state = next_state
    return torch.stack(buffer)

def random_policy(state):
    return 1 if torch.rand(1).item() > 0.5 else 0

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    tensor = collect_state_action_data(env, random_policy, num_episodes=200)
    torch.save(tensor, "state_action_dataset.pt")
    print("Saved state_action_dataset.pt")
