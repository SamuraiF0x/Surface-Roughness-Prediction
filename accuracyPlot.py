import matplotlib.pyplot as plt


def accuracyPlot(history, metric):
    plt.rcParams["font.size"] = 16

    plt.plot(
        history.history[metric],
        label=(f"Training {metric}"),
        linestyle="dashed",
        linewidth=2,
    )
    plt.plot(
        history.history[f"val_{metric}"],
        label=(f"Validation {metric}"),
        linestyle="solid",
        linewidth=2,
    )

    plt.title(f"Model {metric}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss" if metric == "loss" else "Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
