import glob
import os
import sys
import pathlib
import re
from typing import Iterable
import torch
import torchmetrics

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd

import aido

matplotlib.use("agg")


class CaloOptPlotting:

    def __init__(self, results_dir: str | os.PathLike) -> None:
        self.results_dir = results_dir
        self.reco_output_paths = {}
        for file_name in glob.glob(f"{results_dir}/task_outputs/iteration=*/validation=False/reco_output_df"):
            iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
            self.reco_output_paths[iteration] = file_name
        self.parameter_paths = {
            iteration : f"{results_dir}/parameters/param_dict_iter_{iteration}.json"
            for iteration in self.reco_output_paths.keys()
        }

    @staticmethod
    def mplstyle() -> None:
        plt.style.use(pathlib.Path(__file__).parent / "aido.mplstyle")

    @classmethod
    def add_plot_header(cls, ax: plt.Axes) -> plt.Axes:
        plt.text(
            0.0, 1.06,
            "AIDO",
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left'
        )
        plt.text(
            0.125, 1.06,
            "Detector Optimization",
            transform=ax.transAxes, fontsize=14, style='italic', va='top', ha='left'
        )
        #plt.text(
        #    0.015, 0.98,
        #    "Sampling Calorimeter\n"
        #    "50% photons and 50% pions\n"
        #    r"$20 \times 400$" + " MC Events / Iteration\n"
        #    r"$E_\text{true}=[1, 20]$" + " GeV",
        #    transform=ax.transAxes, va='top', ha='left'
        #)  # Adjust the text as fitting
        return ax

    def plot(self, parameter_dict: aido.SimulationParameterDictionary | None = None) -> None:

        def plot_reco_loss(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)
            e_rec: pd.Series = df["Loss"]["Reco_loss"]
            plt.hist(
                e_rec,
                bins=bins,
                color=color,
                histtype="step",
                label=label,
                linewidth=1,
                zorder=iteration
            )

        def plot_class_loss(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)
            e_cls: pd.Series = df["Loss"]["Class_loss"]
            plt.hist(
                e_cls,
                bins=bins,
                color=color,
                histtype="step",
                label=label,
                linewidth=1,
                zorder=iteration
            )

        def plot_reco_loss_all() -> None:
            sampled_iterations = [0, 10, 20, 200]
            cmap = plt.get_cmap('coolwarm', len(sampled_iterations))
            fig, ax = plt.subplots()
            bins = np.linspace(0, 10, 100 + 1)

            for iteration in sampled_iterations:
                if iteration in self.reco_output_paths.keys():
                    plot_reco_loss(
                        iteration=iteration,
                        file_name=self.reco_output_paths[iteration],
                        bins=bins,
                        color=cmap(iteration),
                        label=(f"Iteration {iteration:3d}"),
                    )

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax = self.add_plot_header(ax)
            ax.legend(handles, labels)
            plt.yscale("log")
            plt.xlim(bins[0], bins[-1])
            plt.ylim(1, 5000)
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
            plt.xlabel("Reconstruction Loss [GeV]")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/reco_loss_all"))
            plt.close()

        def plot_class_loss_all() -> None:
            sampled_iterations = [0, 10, 20, 200]
            cmap = plt.get_cmap('coolwarm', len(sampled_iterations))
            fig, ax = plt.subplots()
            bins = np.linspace(0, 10, 100 + 1)

            for iteration in sampled_iterations:
                if iteration in self.reco_output_paths.keys():
                    plot_class_loss(
                        iteration=iteration,
                        file_name=self.reco_output_paths[iteration],
                        bins=bins,
                        color=cmap(iteration),
                        label=(f"Iteration {iteration:3d}"),
                    )

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax = self.add_plot_header(ax)
            ax.legend(handles, labels)
            plt.yscale("log")
            plt.xlim(bins[0], bins[-1])
            plt.ylim(1, 5000)
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
            plt.xlabel("Classification Loss [GeV]")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/reco_class_all"))
            plt.close()


        def plot_energy_resolution_single(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)
            e_rec: pd.Series = df["Reconstructed"]["true_energy"] - df["Targets"]["true_energy"]
            e_rec = e_rec / np.sqrt(df["Targets"]["true_energy"])
            fwhm = aido.Plotting.FWHM(bins, np.histogram(e_rec,bins)[0])
            plt.hist(
                e_rec,
                bins=bins,
                color=color,
                histtype="step",
                label=label + f' (FWHM = {fwhm.width:.3f} GeV$^{{1/2}}$)',
                linewidth=1,
                zorder=iteration
            )

        def plot_energy_resolution_all() -> None:
            sampled_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            sampled_iterations = [0, 5, 10, 15, 20, 25, 30]
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(sampled_iterations)))
            fig, ax = plt.subplots()
            bins = np.linspace(-10, 10, 50 + 1)

            for i,iteration in enumerate(sampled_iterations):
                if iteration in self.reco_output_paths.keys():
                    plot_energy_resolution_single(
                        iteration=iteration,
                        file_name=self.reco_output_paths[iteration],
                        bins=bins,
                        color=colors[i],
                        label=f"Iteration {iteration:3d}",
                    )

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax = self.add_plot_header(ax)
            ax.legend(handles, labels, fontsize=12)
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV" + r"$^{1/2}$" + ")")
            plt.xlabel(r"$(E_\text{rec} - E_\text{true}) / E_\text{true}^{1/2}\, \left[ \text{GeV}^{1/2} \right]$")
            ymin, ymax = plt.ylim()
            plt.ylim(1e-1,ymax*20)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_all"))
            plt.close()

        def plot_fpr_tpr(fprs,tprs,iterations):
            fig, axs = plt.subplots(ncols=len(fprs.keys()),figsize=(6*len(fprs.keys()),5))
            if not isinstance(axs,np.ndarray):
                axs = np.array([axs])
            colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(fprs)))
            for i,name in enumerate(fprs.keys()):
                for j,(fpr,tpr) in enumerate(zip(fprs[name],tprs[name])):
                    axs[i].plot(tpr,fpr,label=f'Iteration {iterations[j]:3d} (AUC = {auc(fpr,tpr):.5f})')
                axs[i].plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle='dashed',color='grey',label='Random classifier')
                axs[i].set_xlabel('TPR')
                axs[i].set_ylabel('FPR')
                axs[i].set_title(f'{name} ROC curve')
                axs[i].legend()
                axs[i].set_xlim(0,1)
                axs[i].set_ylim(0,1)
                axs[i].set_yscale('symlog',linthresh=1e-2)
            return fig


        def plot_classification_roc_all() -> None:
            sampled_iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            sampled_iterations = [0, 5, 10, 15, 20, 25, 30]
            fprs = {}
            tprs = {}
            for iteration in sampled_iterations:
                if iteration in self.reco_output_paths.keys():
                    df = pd.read_parquet(self.reco_output_paths[iteration])
                    columns = list(df["Classes"].columns)
                    names = [col.replace('contains:','') for col in columns]
                    for i in range(len(columns)):
                        fpr, tpr, _= roc_curve(
                            df["Classes"][columns[i]].values,
                            df["Reconstructed"][f"true_logits_{i}"].values,
                        )
                        if names[i] in fprs.keys():
                            fprs[names[i]].append(fpr)
                        else:
                            fprs[names[i]] = [fpr]
                        if names[i] in tprs.keys():
                            tprs[names[i]].append(tpr)
                        else:
                            tprs[names[i]] = [tpr]

            fig = plot_fpr_tpr(fprs,tprs,sampled_iterations)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/classification_roc_all"))
            plt.close()

        def plot_CM_all() -> None:
            sampled_iterations = [0, 5, 10, 15, 20]
            cms = []
            for iteration in sampled_iterations:
                if iteration in self.reco_output_paths.keys():
                    df = pd.read_parquet(self.reco_output_paths[iteration])
                    columns = list(df["Classes"].columns)
                    names = [col.replace('contains:','') for col in columns]
                    labels = df["Classes"][columns].values.astype(np.float32)
                    preds = df["Reconstructed"][[f"true_logits_{i}" for i in range(len(columns))]].values
                    if np.all(labels.sum(axis=1)==1):
                        cms.append(
                            confusion_matrix(
                                labels.argmax(axis=1),
                                preds.argmax(axis=1),
                                normalize = 'true',
                            )
                        )
            if len(cms) == 0:
                print ('Not a multiclassification task')
                return

            fig,axs = plt.subplots(ncols=len(cms),figsize=(6*len(cms),5))
            if not isinstance(axs,np.ndarray):
                axs = np.array([axs])

            for it,cm,ax in zip(sampled_iterations,cms,axs):
                disp = ConfusionMatrixDisplay(cm,display_labels=names)
                disp.plot(
                    ax = ax,
                    colorbar = False,
                    cmap = 'Blues',
                    im_kw = {'norm': matplotlib.colors.Normalize(vmin=0, vmax=1)},
                )
                ax.set_title(f'Iteration {it}')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/classification_CM_all"))
            plt.close()

        def plot_CM_evolution() -> None:
            iterations = sorted(list(self.reco_output_paths.keys()))
            cms = []
            for iteration in iterations:
                if iteration in self.reco_output_paths.keys():
                    df = pd.read_parquet(self.reco_output_paths[iteration])
                    columns = list(df["Classes"].columns)
                    names = [col.replace('contains:','') for col in columns]
                    labels = df["Classes"][columns].values.astype(np.float32)
                    preds = df["Reconstructed"][[f"true_logits_{i}" for i in range(len(columns))]].values
                    if np.all(labels.sum(axis=1)==1):
                        cms.append(
                            confusion_matrix(
                                labels.argmax(axis=1),
                                preds.argmax(axis=1),
                                normalize = 'true',
                            )
                        )

            if len(cms) == 0:
                print ('Not a multiclassification task')
                return
            order = np.array(iterations).argsort()

            for i,name in enumerate(names):
                evolution = np.concatenate(
                    [
                        cms[j][i].reshape(-1,1)
                        for j in order
                    ],
                    axis = 1
                )
                fig,ax = plt.subplots(figsize=(evolution.shape[1],5))
                plt.subplots_adjust(left=0.1,right=0.9,bottom=0.05,top=0.95)
                im = ax.imshow(evolution, cmap='Blues', vmin=0., vmax=1.)
                cbar = fig.colorbar(im, ax=ax, label=f'P(class|{name} event)', shrink=0.6, aspect=20, pad=0.02)
                ax.set_title(f'True class {name}',fontsize=18)
                ax.set_xticks(np.arange(evolution.shape[1]))
                ax.set_yticks(np.arange(evolution.shape[0]))
                ax.set_yticklabels(names,fontsize=16)
                ax.set_xlabel('Iterations',fontsize=16)
                ax.set_ylabel('Predicted classes',fontsize=16)

                plt.savefig(os.path.join(self.results_dir, f"plots/classification_CM_evolution_{name}"), bbox_inches='tight')
                plt.close()




        def plot_energy_resolution_first_and_last() -> None:
            fig, ax = plt.subplots()
            ax = self.add_plot_header(ax)
            cmap = plt.get_cmap('coolwarm', len(self.reco_output_paths))
            bins = np.linspace(-50, 50, 100 + 1)
            iterations = []
            for iteration in [min(self.reco_output_paths.keys()),max(self.reco_output_paths.keys())]:
                df = pd.read_parquet(self.reco_output_paths[iteration])
                e_rec = (df["Targets"]["true_energy"] - df["Reconstructed"]["true_energy"])
                e_rec_binned, *_ = plt.hist(
                    e_rec,
                    bins=bins,
                    color=cmap(iteration),
                    histtype="step",
                    label=f"Iteration {iteration:3d}",
                )
                ax = aido.Plotting.FWHM(bins, e_rec_binned).add_to_axis(ax)

            plt.legend()
            plt.xlim(-50, 50)
            ymin, ymax = plt.ylim()
            plt.ylim(1e-1,ymax*20)
            plt.yscale('log')
            plt.xlabel(r"Energy Resolution $E_{\text{true}} - E_{\text{rec}}$ [GeV]")
            plt.ylabel(f"Counts {(bins[1] - bins[0]):.2f}")
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_first_and_last"))
            plt.close()

        def plot_classification_first_and_last() -> None:
            fprs = {}
            tprs = {}
            iterations = []
            for iteration in [min(self.reco_output_paths.keys()),max(self.reco_output_paths.keys())]:
                iterations.append(iteration)
                df = pd.read_parquet(self.reco_output_paths[iteration])
                columns = list(df["Classes"].columns)
                names = [col.replace('contains:','') for col in columns]
                for i in range(len(columns)):
                    fpr, tpr, _= roc_curve(
                        df["Classes"][columns[i]].values,
                        df["Reconstructed"][f"true_logits_{i}"].values,
                    )
                    if names[i] in fprs.keys():
                        fprs[names[i]].append(fpr)
                    else:
                        fprs[names[i]] = [fpr]
                    if names[i] in tprs.keys():
                        tprs[names[i]].append(tpr)
                    else:
                        tprs[names[i]] = [tpr]
            fig = plot_fpr_tpr(fprs,tprs,iterations)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/classification_roc_first_and_last"))
            plt.close()

        def plot_classification_metrics_evolution():
            iterations = sorted(list(self.reco_output_paths.keys()))
            metric_functions = {
                'Accuracy': torchmetrics.classification.BinaryAccuracy(),
                'AUC' : torchmetrics.classification.BinaryAUROC(),
                'Recall' : torchmetrics.classification.BinaryRecall(),
                'F1' : torchmetrics.classification.BinaryF1Score(),
            }
            metrics_values = {}
            for iteration in iterations:
                if iteration in self.reco_output_paths.keys():
                    df = pd.read_parquet(self.reco_output_paths[iteration])
                    columns = list(df["Classes"].columns)
                    names = [col.replace('contains:','') for col in columns]
                    for i in range(len(columns)):
                        if names[i] not in metrics_values.keys():
                            metrics_values[names[i]] = {}
                        for label,func in metric_functions.items():
                            if label not in metrics_values[names[i]]:
                                metrics_values[names[i]][label] = []
                            metrics_values[names[i]][label].append(
                                func(
                                    torch.nn.functional.sigmoid(torch.tensor(df["Reconstructed"][f"true_logits_{i}"].values)),
                                    torch.tensor(df["Classes"][columns[i]].values),
                                )
                            )

            fig, axs = plt.subplots(ncols=len(metrics_values),figsize=(6*len(metrics_values.keys()),5))
            iterations = np.array(iterations)
            order = iterations.argsort()
            if not isinstance(axs,np.ndarray):
                axs = np.array([axs])
            for i,name in enumerate(metrics_values.keys()):
                metrics = metrics_values[name]
                for label,values in metrics.items():
                    values = np.array(values)
                    axs[i].plot(
                        iterations[order],
                        values[order],
                        label = label,
                    )
                axs[i].set_xlabel('Iterations')
                axs[i].set_xlim(iterations.min(),iterations.max())
                axs[i].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/classification_metrics_evolution"))
            plt.close()



        def plot_energy_resolution_evolution():
            iterations = sorted(list(self.reco_output_paths.keys()))
            fwhms = []
            metric_functions = {
                'MAE': torchmetrics.regression.MeanAbsoluteError(),
                'MSE': torchmetrics.regression.MeanSquaredError(),
                'MAPE': torchmetrics.regression.MeanAbsolutePercentageError(),
                'R2': torchmetrics.regression.R2Score(),
            }
            metrics_values = {label:[] for label in metric_functions.keys()}
            for iteration in iterations:
                df = pd.read_parquet(self.reco_output_paths[iteration])
                e_rec = (df["Targets"]["true_energy"] - df["Reconstructed"]["true_energy"]) / np.sqrt(df["Targets"]["true_energy"])
                bins = np.linspace(e_rec.min(),e_rec.max(),40)
                fwhm = aido.Plotting.FWHM(bins, np.histogram(e_rec,bins)[0])
                fwhms.append(fwhm.width)
                e_true = torch.tensor(df["Targets"]["true_energy"].values)
                e_reco = torch.tensor(df["Reconstructed"]["true_energy"].values)
                for label, func in metric_functions.items():
                    metrics_values[label].append(func(e_reco,e_true))
            fig,axs = plt.subplots(ncols=len(metric_functions)+1, figsize=(6*len(metric_functions)+1,5))
            iterations = np.array(iterations)
            fwhms = np.array(fwhms)
            order = iterations.argsort()
            iterations = iterations[order]
            fwhms = fwhms[order]
            axs[0].plot(
                iterations,
                fwhms,
            )
            axs[0].set_xlabel('Iterations')
            axs[0].set_ylabel(r"FWHM $(E_\text{rec} - E_\text{true}) / E_\text{true}^{1/2}\, \left[ \text{GeV}^{1/2} \right]$")
            axs[0].set_xlim(iterations.min(),iterations.max())
            for i,(label,metrics) in enumerate(metrics_values.items()):
                metrics = np.array(metrics)[order]
                axs[i+1].plot(
                    iterations,
                    metrics,
                )
                axs[i+1].set_xlabel('Iterations')
                axs[i+1].set_ylabel(label)
                axs[i+1].set_xlim(iterations.min(),iterations.max())
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_evolution"))
            plt.close()



        def plot_calorimeter_sideview(
            add_legend: bool = None,
        ) -> None:
            df_list = []
            df_materials_list = []
            parameter_dir = os.path.join(self.results_dir, "parameters/")

            for file_name in os.listdir(parameter_dir):
                param_dict = aido.SimulationParameterDictionary.from_json(parameter_dir + file_name)
                df_list.append(pd.DataFrame(
                    param_dict.get_current_values(format="dict", types="continuous"),
                    index=[param_dict.iteration],
                ))
                df_materials = pd.DataFrame(param_dict.get_probabilities()).drop(index=0)
                df_materials.index = [param_dict.iteration]
                df_materials_list.append(df_materials)

            df: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
            df_materials: pd.DataFrame = pd.concat(df_materials_list, axis=0).sort_index()
            df_materials.columns = df.columns

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax = self.add_plot_header(ax)
            absorber_cmap = mcolors.LinearSegmentedColormap.from_list("blue_grey", ["blue", "grey"])
            scintillator_cmap = plt.get_cmap("spring")

            def get_color(label: str, prob: Iterable):
                if "absorber" in label:
                    return absorber_cmap(prob)
                if "scintillator" in label:
                    return scintillator_cmap(prob)
                else:
                    return "white"

            for i in df.index:
                bottom = 0
                plt.gca().set_prop_cycle(None)

                for column in df.columns:
                    plt.bar(
                        i,
                        df[column][i],
                        bottom=bottom,
                        color=get_color(column, df_materials[column][i]),
                        width=1,
                        align="edge",
                        label=column.replace("_", " ").removeprefix("thickness ").capitalize(),
                    )
                    bottom += df[column][i]

            if add_legend:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper right")

            plt.ylabel("Longitudinal Calorimeter Composition [cm]")
            plt.xlabel("Iteration")
            plt.xlim(0, len(df))
            plt.ylim(0, 110)
            ax = self.add_plot_header(ax)
            cbar_absorber = plt.cm.ScalarMappable(cmap=absorber_cmap)
            cbar_absorber.set_array([])
            cbar1 = plt.colorbar(
                cbar_absorber,
                ax=ax,
                fraction=0.04,
                location="right",
            )
            cbar1.ax.invert_yaxis()  # Invert so Fe is high and Pb is low
            cbar1.ax.set_yticks([0, 1], labels=['Pb', 'Fe'], rotation=90, va='center')  # Custom ticks

            # Add colormap for "scintillator" with labels "Polystyrene" and "PbWO4"
            cbar_scintillator = plt.cm.ScalarMappable(cmap=scintillator_cmap)
            cbar_scintillator.set_array([])
            cbar2 = plt.colorbar(
                cbar_scintillator,
                ax=ax,
                fraction=0.04,
                location="right",
            )
            cbar2.ax.invert_yaxis()
            cbar2.ax.set_yticks([0, 1], labels=['PbWO4', 'Polystyrene'], rotation=90, va='center')  # Custom ticks
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/calorimeter_sideview"), dpi=500)
            plt.close()

        def mask_parameters(df,param_dict):
            masks = []
            for param_name, param_value in param_dict.get_current_values(format="dict", types="continuous").items():
                masks.append(
                    (abs(df['Parameters'][param_name] - param_value) < 1e-5).values
                )
            for param_name, param_value in param_dict.get_current_values(format="dict", types="discrete").items():
                masks.append(
                    df['Parameters'][f'{param_name}_{param_value}'] == 1
                )
            idx = np.where(np.logical_and.reduce(masks))[0]
            assert len(idx) > 0
            return idx

        def plot_loss_evolutions(use_checkpoint: bool = False) -> None:
            reco_losses = []
            class_losses = []
            reco_losses_best = []
            class_losses_best = []
            iterations = sorted(list(self.reco_output_paths.keys()))

            for iteration in iterations:
                df = pd.read_parquet(self.reco_output_paths[iteration])
                param_dict = aido.SimulationParameterDictionary.from_json(self.parameter_paths[iteration])
                idx = mask_parameters(df,param_dict)
                reco_losses.append(np.mean(df["Loss"]["Reco_loss"]))
                class_losses.append(np.mean(df["Loss"]["Class_loss"]))
                reco_losses_best.append(np.mean(df["Loss"]["Reco_loss"][idx]))
                class_losses_best.append(np.mean(df["Loss"]["Class_loss"][idx]))

            iterations = np.array(iterations)
            order = iterations.argsort()
            iterations = iterations[order]
            reco_losses = np.array(reco_losses)[order]
            class_losses = np.array(class_losses)[order]
            reco_losses_best = np.array(reco_losses_best)[order]
            class_losses_best = np.array(class_losses_best)[order]

            df_loss: pd.DataFrame = aido.Plotting.optimizer_loss(results_dir=self.results_dir)
            df_loss = df_loss[["Scaled Epoch", "Loss"]]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax = self.add_plot_header(ax)
            plt.plot(
                iterations,
                reco_losses,
                color = 'royalblue',
                linestyle = 'solid',
                label="Mean Reconstruction Loss " + r"($\mathcal{L}_\text{reco}$)",
            )
            plt.plot(
                iterations,
                reco_losses_best,
                color = 'royalblue',
                linestyle = 'dashed',
            )
            plt.plot(
                iterations,
                class_losses,
                color = 'red',
                linestyle = 'solid',
                label="Mean Classification Loss " + r"($\mathcal{L}_\text{class}$)",
            )
            plt.plot(
                iterations,
                class_losses_best,
                color = 'red',
                linestyle = 'dashed',
            )
            plt.plot(
                df_loss["Scaled Epoch"],
                df_loss["Loss"],
                color = 'green',
                label="Optimizer Loss " + r"($\mathcal{L}'$)"
            )
            plt.plot(
                [],[],
                color = 'black',
                linestyle = 'solid',
                label = 'Sampled parameters',
            )
            plt.plot(
                [],[],
                color = 'black',
                linestyle = 'dashed',
                label = 'Best parameters',
            )
            plt.legend(loc='upper right',fontsize=10)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.xlim(
                min(iterations.min(),df_loss["Scaled Epoch"].min()),
                max(iterations.max(),df_loss["Scaled Epoch"].max()),
            )
            plt.ylim(
                min(reco_losses.min(),class_losses.min(),df_loss["Loss"].min())/2,
                max(reco_losses.max(),class_losses.max(),df_loss["Loss"].max())*2,
            )
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/loss_evolution.png"))
            plt.close()

        def plot_constraints() -> None:
            def cost(parameter_dict: aido.SimulationParameterDictionary) -> float:
                cost = 0.0
                for i in range(parameter_dict[f"num_layers"].current_value):
                    for name in ["absorber", "scintillator"]:
                        cost += (
                            parameter_dict[f"thickness_{name}:{i}"].current_value
                            * np.array(parameter_dict[f"material_{name}:{i}"].weighted_cost)
                        )
                return cost

            cost_list = []
            for i in range(len(self.reco_output_paths)):
                sim_param_dict = aido.SimulationParameterDictionary.from_json(
                    f"{self.results_dir}/parameters/param_dict_iter_{i}.json"
                )
                cost_item = cost(sim_param_dict)
                cost_list.append(cost_item)

            fig, ax = plt.subplots(figsize=(7, 5))
            plt.plot(cost_list)
            plt.xlabel("Iteration")
            plt.ylabel("Cost [EUR]")
            plt.xlim(0, len(cost_list) + 1)
            plt.ylim(0,)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/cost_constraints.png"))
            plt.close()

        if len(self.reco_output_paths) <= 1:
            print(f"No task outputs found in '{self.results_dir}/task_outputs/'. Skipping plotting.")
            return None


        plot_energy_resolution_all()
        plot_energy_resolution_first_and_last()
        plot_energy_resolution_evolution()
        plot_classification_roc_all()
        plot_classification_first_and_last()
        plot_CM_evolution()
        plot_CM_all()
        plot_classification_metrics_evolution()
        plot_reco_loss_all()
        plot_class_loss_all()
        plot_loss_evolutions()
        plot_calorimeter_sideview()
        plot_constraints()
        plt.close("all")
        print (f'Plots saved in {self.results_dir}')
        return None


if __name__ == "__main__":
    results_dir: str = sys.argv[1]

    plotter = CaloOptPlotting(results_dir)
    plotter.mplstyle()
    plotter.plot()
