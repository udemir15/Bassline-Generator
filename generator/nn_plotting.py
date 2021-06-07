import matplotlib.pyplot as plt


def plot_loss_history(hist, save_fig=False):
    plt.figure(figsize=(20, 6))

    plt.plot(hist.history['loss'], label='training')
    plt.plot(hist.history['val_loss'], label='testing')
    plt.legend()

    if save_fig:
        plt.savefig(f'figures/{name}', dpi=400)
