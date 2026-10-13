import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GateVisualizer:
    """
    Visualization tools for phenotype-conditioned gates.
    """

    @staticmethod
    def plot_gate_distributions(gate_values, save_path=None):
        """
        Plot distributions of gate values for each propagation step.

        Args:
            gate_values: Dict of gate tensors (B, 1) from model.get_gate_values()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, len(gate_values), figsize=(15, 3))
        if len(gate_values) == 1:
            axes = [axes]

        for idx, (gate_name, values) in enumerate(gate_values.items()):
            values_np = values.squeeze().cpu().numpy()
            axes[idx].hist(values_np, bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{gate_name} gate')
            axes[idx].set_xlabel('Gate value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(0.5, color='red', linestyle='--', label='0.5 threshold')
            axes[idx].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def export_gate_values_csv(gate_values, sample_ids, save_path):
        """
        Export gate values to CSV for downstream analysis.

        Args:
            gate_values: Dict of gate tensors
            sample_ids: List of sample IDs
            save_path: Path to save CSV
        """
        data = {'sample_id': sample_ids}
        for gate_name, values in gate_values.items():
            data[gate_name] = values.squeeze().cpu().numpy()

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Gate values exported to {save_path}")

    @staticmethod
    def plot_gate_evolution(gate_history, save_path=None):
        """
        Plot how gate values evolve during training.

        Args:
            gate_history: Dict of lists, each list containing gate stats over epochs
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot mean gate values
        for gate_name, history in gate_history.items():
            epochs = range(len(history))
            means = [h['mean'] for h in history]
            axes[0].plot(epochs, means, label=gate_name, marker='o')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean gate value')
        axes[0].set_title('Gate Value Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot sparsity (% gates < 0.1)
        for gate_name, history in gate_history.items():
            epochs = range(len(history))
            sparsity = [h['sparsity'] for h in history]
            axes[1].plot(epochs, sparsity, label=gate_name, marker='o')

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Sparsity (% < 0.1)')
        axes[1].set_title('Gate Sparsity Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_gate_heatmap(gate_values, phenotype_names=None, save_path=None):
        """
        Plot heatmap of gate values across samples.

        Args:
            gate_values: Dict of gate tensors
            phenotype_names: Optional list of phenotype names
            save_path: Optional path to save figure
        """
        # Convert gate values to numpy array
        gate_names = list(gate_values.keys())
        gate_array = np.stack([gate_values[name].squeeze().cpu().numpy()
                               for name in gate_names], axis=1)  # (B, num_gates)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(6, gate_array.shape[0] * 0.3)))
        sns.heatmap(gate_array, cmap='RdYlGn', vmin=0, vmax=1,
                    xticklabels=gate_names, yticklabels=phenotype_names if phenotype_names else False,
                    cbar_kws={'label': 'Gate value'}, ax=ax)
        ax.set_xlabel('Propagation step')
        ax.set_ylabel('Samples')
        ax.set_title('Gate Values Heatmap')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
