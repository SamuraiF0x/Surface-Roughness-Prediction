import matplotlib.pyplot as plt


def accuracyPlot(history):
    plt.plot(history.history["accuracy"], label="Training accuracy", linestyle="dashed")
    plt.plot(
        history.history["val_accuracy"], label="Validation accuracy", linestyle="solid"
    )
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
