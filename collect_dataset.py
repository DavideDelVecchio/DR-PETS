import torch
import gym

def collect_state_action_transitions(env, policy, num_episodes=100):
    transitions = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # One-hot encode action
            action_onehot = [1.0 if i == action else 0.0 for i in range(env.action_space.n)]
            transition = list(state) + action_onehot + list(next_state)
            transitions.append(torch.tensor(transition, dtype=torch.float32))
            state = next_state
    return torch.stack(transitions)

def random_policy(state):
    return torch.randint(0, 2, ()).item()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    transitions = collect_state_action_transitions(env, random_policy, num_episodes=200)
    torch.save(transitions, "state_action_dataset.pt")
    print("Saved (state, action_onehot, next_state) transitions to state_action_dataset.pt")