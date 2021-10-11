import matplotlib.pyplot as plt

def plot_pred_curve(pred, label):
    # Plotting
    plt.plot(pred[0, :, 0], '-b')
    plt.plot(pred[0, :, 1], '-b')
    plt.plot(pred[0, :, 2], '-b')

    plt.plot(label[0, :, 0], '-g')
    plt.plot(label[0, :, 1], '-g')
    plt.plot(label[0, :, 2], '-g')
    plt.show()
