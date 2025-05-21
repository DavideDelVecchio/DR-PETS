import matplotlib.pyplot as plt

def plot_log_likelihoods(log_probs, title="State-Action Log-Likelihoods"):
    plt.figure()
    plt.plot(log_probs.numpy())
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.show()

def compare_trajectories(log_probs_list, labels):
    plt.figure()
    for log_probs, label in zip(log_probs_list, labels):
        plt.plot(log_probs.numpy(), label=label)
    plt.title("Comparison of Log-Likelihoods")
    plt.xlabel("Time step")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.grid(True)
    plt.show()