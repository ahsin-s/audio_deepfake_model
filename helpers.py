import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(tp, fp, fn, tn, axis_labels, title: str, output_path: str):
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=axis_labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(title)
    plt.savefig(output_path)
    plt.close()  # Close the figure to avoid memory issues
    print("Confusion Matrix plot saved")
    return True