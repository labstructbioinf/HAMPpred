from matplotlib import pyplot as plt


def reg_plot(tr, pr, title='Predictions for data set', ylabel='True'):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.scatter(pr, tr, color='green', s=25)
    ax.set_ylim(-26, 26)
    ax.set_xlim(-26, 26)
    ax.plot([0, 1], [0, 1], '--', transform=ax.transAxes, color='grey')
    ax.set_xlabel('Predicted', size=15)
    ax.set_ylabel(ylabel, size=15)
    ax.set_title(title)
    return ax
