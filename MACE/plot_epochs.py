import matplotlib.pyplot as plt
import pandas as pd
import json

def main():
    with open('Results/mace1_25iter_train.txt') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    opt_data = df[df['mode'] == 'opt']  # Filter out optimization data
    eval_data = df[df['mode'] == 'eval']  # Filter evaluation data

    # 1. Plot Loss vs. Epoch (for "opt" mode)
    plt.figure(figsize=(8, 6))
    #plt.plot(opt_data['epoch'], opt_data['loss'], 'o-', label='Loss')
    plt.plot(eval_data['epoch'], eval_data['loss'], 'o-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.yscale('log')
    plt.xlim(0,200)
    #plt.ylim(0, 0.0008)
    plt.title('Loss vs. Epoch')
    plt.grid(True)
    plt.savefig("Figures/loss_vs_epoch.png")
    plt.show()

    # 2. Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(opt_data['loss'], 'o-', label='Loss')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss')
    plt.grid(True)
    plt.savefig("Figures/loss.png")
    plt.show()

    # 3. Plot Evaluation Metrics over Epoch (MAE and RMSE for Energy and Forces)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot MAE for Energy and Forces
    ax1 = axs[0]
    ax1.plot(eval_data['epoch'], eval_data['mae_e'], 'o-', color='C0', label='MAE Energy')
    ax1.set_xlabel('Epoch', color='black')
    ax1.set_ylabel('MAE Energy', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(0, 200)
    ax1.grid(True)

    # Secondary axis for MAE Forces
    ax2 = ax1.twinx()
    ax2.plot(eval_data['epoch'], eval_data['mae_f'], 'o-', color='C1', label='MAE Forces')
    ax2.set_ylabel('MAE Forces', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax1.set_title('MAE for Energy and Forces')

    # Legend for both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot RMSE for Energy and Forces
    ax3 = axs[1]
    ax3.plot(eval_data['epoch'], eval_data['rmse_e'], 'o-', color='C0', label='RMSE Energy')
    ax3.set_xlabel('Epoch', color='black')
    ax3.set_ylabel('RMSE Energy', color='black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_xlim(0, 200)
    ax3.grid(True)

    # Secondary axis for RMSE Forces
    ax4 = ax3.twinx()
    ax4.plot(eval_data['epoch'], eval_data['rmse_f'], 'o-', color='C1', label='RMSE Forces')
    ax4.set_ylabel('RMSE Forces', color='black')
    ax4.tick_params(axis='y', labelcolor='black')
    ax3.set_title('RMSE for Energy and Forces')

    # Legend for both y-axes
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig("Figures/training_errors.png")
    plt.show()

if __name__ == "__main__":
    main()