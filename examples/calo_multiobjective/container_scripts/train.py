# flake8: noqa: E402
import os
import pickle
import pathlib
import sys
from typing import Optional, Union
from sklearn.model_selection import train_test_split


import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from utils import LossPlotting, EarlyStopping
from reconstruction.dataset import ReconstructionDataset
from reconstruction.model import Reconstruction
from reconstruction.validation_reconstruction import ReconstructionValidation
from classification.dataset import ClassificationDataset
from classification.dataset import FlipAugmentation, RotationAugmentation, ShiftAugmentation
from classification.model import Classification, Classification, Classification2
from classification.validation_classification import ClassificationValidation


config = {
    "reconstruction": {
        "n_epochs_pre": 50,
        "n_epochs_main": 50,
        "batch_size": 1024,
        "early_stopping": 50,
        "lr_pre": 0.01,
        "lr_main": [
            0.005,
            0.001,
            0.0005
        ],
        "targets": [
            "true_energy"
        ]
    },
    "classification": {
        "n_epochs_pre": 25,
        "n_epochs_main": 25,
        "batch_size": 1024,
        "early_stopping": 25,
        "lr_pre": 0.001,
        "lr_main": [
            0.0005,
            0.0001,
            5e-05
        ],
        "classes": [
            "contains:e+",
            "contains:gamma",
            #"contains:pi+",
            #"contains:proton"
        ],
        "multiclass": True,
        "weight": [
            1.0,
            1.0,
            #5.0,
            #5.0
        ]
    }
}

def pre_train(
    model: Reconstruction,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    n_epochs: int,
    batch_size: int,
    lr: float,
    plotter: LossPlotting,
    early_stopping: EarlyStopping = None,
):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalized ones,
    but also requires adjustments to the surrogate dataset
    """

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training")
    model.train_model(
        train_dataset,
        valid_dataset,
        batch_size = batch_size,
        n_epochs = n_epochs,
        lr = lr,
        plotter = plotter,
        early_stopping = early_stopping,
    )

    model.to('cpu')

def split(reco_data: dict):
    idx = np.arange(len(reco_data["Inputs"]["sensor_energy"]))
    train_idx, valid_idx = train_test_split(idx, test_size=0.4, shuffle=True)
    valid_idx, test_idx = train_test_split(valid_idx, test_size=0.5, shuffle=True)

    train_data = {
        "Inputs": {key: val[train_idx] for key, val in reco_data["Inputs"].items()},
        "Targets": {key: val[train_idx] for key, val in reco_data["Targets"].items()},
        "Classes": {key: val[train_idx] for key, val in reco_data["Classes"].items()},
    }
    val_data = {
        "Inputs": {key: val[valid_idx] for key, val in reco_data["Inputs"].items()},
        "Targets": {key: val[valid_idx] for key, val in reco_data["Targets"].items()},
        "Classes": {key: val[valid_idx] for key, val in reco_data["Classes"].items()},
    }

    test_data = {
        "Inputs": {key: val[test_idx] for key, val in reco_data["Inputs"].items()},
        "Targets": {key: val[test_idx] for key, val in reco_data["Targets"].items()},
        "Classes": {key: val[test_idx] for key, val in reco_data["Classes"].items()},
    }
    
    return train_data, val_data, test_data

def train(
    input_mf_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    model_path: Union[str, os.PathLike],
    results_dir: Optional[Union[str, os.PathLike]] = None,
):  
    #iteration = int(re.search(r"iteration=(\d+)", input_mf_path).group(1))
    print (f'Loading {input_mf_path}')
    with open(input_mf_path,'rb') as f:
        reco_data = pickle.load(f)

    print (f'Simulation dataframe : {len(reco_data["Inputs"]["sensor_energy"])} events')
    train_data, valid_data, test_data = split(reco_data)
    print (f'Train dataset : {len(train_data["Inputs"]["sensor_energy"])} / Validation dataset : {len(valid_data["Inputs"]["sensor_energy"])}')

    if isVal:
        pass
        # Below has not been updated with latest changes
        #reco_model: Reconstruction = torch.load(os.path.join(results_dir, "reco_model"))
        #reco_dataset = ReconstructionDataset(simulation_mf, means=reco_model.means, stds=reco_model.stds)

        #validator = ReconstructionValidation(reco_model)
        #output_mf_val = validator.validate(reco_dataset)
        #output_mf_val.to_parquet(output_mf_path)
        #validator.plot(
        #    output_mf_val,
        #    os.path.join(results_dir, "plots", "validation", "reco_model", "on_validationData")
        #)
    else:
        # Reconstruction training:
        reco_model_previous_path = os.path.join(model_path,"reco.pt")
        reco_loss_plotter = LossPlotting()
        reco_early_stopping = EarlyStopping(patience=config["reconstruction"]["early_stopping"])

        if os.path.exists(reco_model_previous_path):
            raise Exception()
            #print ('Loading reco model')
            #reco_model: Reconstruction = torch.load(reco_model_previous_path,weights_only=False)
            #print (reco_model)
            #print("Parameters:", sum([p.numel() for p in reco_model.parameters()]))
            #print ('Creating reco dataset')
            #reco_train_dataset = ReconstructionDataset(
            #    targets = config["reconstruction"]["targets"],
            #    input_data = train_data,
            #    means = reco_model.means,
            #    stds = reco_model.stds,
            #)
            #reco_valid_dataset = ReconstructionDataset(
            #    targets = config["reconstruction"]["targets"],
            #    input_data = valid_data,
            #    means = reco_model.means,
            #    stds = reco_model.stds,
            #)
        else:
            print ('Creating reco dataset')
            reco_train_dataset = ReconstructionDataset(
                targets = config["reconstruction"]["targets"],
                input_data = train_data,
            )
            reco_valid_dataset = ReconstructionDataset(
                targets = config["reconstruction"]["targets"],
                input_data = valid_data,
                means = reco_train_dataset.means,
                stds = reco_train_dataset.stds,
            )
            reco_test_dataset = ReconstructionDataset(
                targets = config["reconstruction"]["targets"],
                input_data = valid_data,
                means = reco_train_dataset.means,
                stds = reco_train_dataset.stds,
            )
            print ('Creating reco model')
            reco_model = Reconstruction(*reco_train_dataset.shape, reco_train_dataset.means, reco_train_dataset.stds)
            print (reco_model)
            print ('Pre-training reco model')
            pre_train(
                model = reco_model,
                train_dataset = reco_train_dataset,
                valid_dataset = reco_valid_dataset,
                n_epochs = config["reconstruction"]["n_epochs_pre"],
                batch_size = config["reconstruction"]["batch_size"],
                lr = config["reconstruction"]["lr_pre"],
                plotter = reco_loss_plotter,
            )

        print ('Train reco model')
        reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        for lr in config["reconstruction"]["lr_main"]:
            reco_model.train_model(
                reco_train_dataset,
                reco_valid_dataset,
                batch_size = config["reconstruction"]["batch_size"],
                n_epochs = config["reconstruction"]["n_epochs_main"],
                lr = lr,
                plotter = reco_loss_plotter,
                early_stopping = reco_early_stopping,
            )
        reco_early_stopping.restore_best_weights(reco_model)

        # Validation #
        torch.save(reco_model, reco_model_previous_path)

        reco_validator = ReconstructionValidation(reco_model)
        train_dict_reco, train_loss_reco = reco_validator.validate(reco_train_dataset)
        val_dict_reco, val_loss_reco = reco_validator.validate(reco_test_dataset)

        del reco_train_dataset
        del reco_valid_dataset
        del reco_model
        
        if results_dir is not None:
            os.makedirs(os.path.join(results_dir,"plots"), exist_ok=True)

        reco_loss_plotter.plot(
            os.path.join(
                results_dir,
                "plots",
                "loss_reco.png",
            )
        )
        reco_validator.plot(
            train_dict_reco,
            os.path.join(
                results_dir,
                "plots",
                f"validation_reco_train.png",
            )
        )
        reco_validator.plot(
            val_dict_reco,
            os.path.join(
                results_dir,
                "plots",
                f"validation_reco_val.png",
            )
        )

        # Classification training:
        class_model_previous_path = os.path.join(model_path, "class.pt")
        class_loss_plotter = LossPlotting()
        class_early_stopping = EarlyStopping(patience=config["classification"]["early_stopping"])
        

        #class_weights = torch.tensor([1.,1.,10.,10.])


        if os.path.exists(class_model_previous_path):
            #print ('Loading class model')
            #class_model: Classification2 = torch.load(class_model_previous_path,weights_only=False)
            #print (class_model)
            #print("Parameters:", sum([p.numel() for p in class_model.parameters()]))
            #print ('Creating class dataset')
            #class_train_dataset = ClassificationDataset(
            #    classes = config["classification"]["classes"],
            #    input_data = train_data,
            #    means = class_model.means,
            #    stds = class_model.stds,
            #    augmentations = [
            #        FlipAugmentation(),
            #        RotationAugmentation(),
            #        ShiftAugmentation(),
            #    ],
            #)
            #class_valid_dataset = ClassificationDataset(
            #    classes = config["classification"]["classes"],
            #    input_data = train_data,
            #    means = class_model.means,
            #    stds = class_model.stds,
            #)
            pass
            
        else:
            print ('Creating class dataset')
            print ('Training')
            class_train_dataset = ClassificationDataset(
                classes = config["classification"]["classes"],
                input_data = train_data,
                augmentations = [
                    FlipAugmentation(),
                    RotationAugmentation(),
                    ShiftAugmentation(),
                ],
            )
            print ('Validation')
            class_valid_dataset = ClassificationDataset(
                classes = config["classification"]["classes"],
                input_data = valid_data,
                means = class_train_dataset.means,
                stds = class_train_dataset.stds,
            )

            class_test_dataset = ClassificationDataset(
                classes = config["classification"]["classes"],
                input_data = test_data,
                means = class_train_dataset.means,
                stds = class_train_dataset.stds,
            )
            print ('Creating class model')
            class_model = Classification2(
                *class_train_dataset.shape,
                initial_means = class_train_dataset.means,
                initial_stds = class_train_dataset.stds,
                multiclass = config["classification"]["multiclass"],
                weight = config["classification"]["weight"],
            )
            print (class_model)
            print ('Pre-training class model')
            pre_train(
                model = class_model,
                train_dataset = class_train_dataset,
                valid_dataset = class_valid_dataset,
                n_epochs = config["classification"]["n_epochs_pre"],
                batch_size = config["classification"]["batch_size"],
                lr = config["classification"]["lr_pre"],
                plotter = class_loss_plotter,
            )

        print ('Train class model')
        class_model.to("cuda" if torch.cuda.is_available() else "cpu")
        for lr in config["classification"]["lr_main"]:
            class_model.train_model(
                class_train_dataset,
                class_valid_dataset,
                batch_size = config["classification"]["batch_size"],
                n_epochs = config["classification"]["n_epochs_main"],
                lr = lr,
                plotter = class_loss_plotter,
                early_stopping = class_early_stopping,
            )
        class_early_stopping.restore_best_weights(class_model)

        # Validation #
        torch.save(class_model, class_model_previous_path)

        class_validator = ClassificationValidation(class_model)
        train_dict_class, train_loss_class = class_validator.validate(class_train_dataset)
        val_dict_class, val_loss_class = class_validator.validate(class_test_dataset)
        print("Finished")

        class_loss_plotter.plot(
            os.path.join(
                results_dir,
                "plots",
                f"loss_class.png",
            )
        )
        class_validator.plot(
            train_dict_class,
            os.path.join(
                results_dir,
                "plots",
                f"validation_class_train.png",
            )
        )
        class_validator.plot(
            val_dict_class,
            os.path.join(
                results_dir,
                "plots",
                f"validation_class_val.png",
            )
        )

        val_losses = np.array([val_loss_reco, val_loss_class])
        np.savetxt(output_df_path, val_losses)

        # Combined mf for surrogate task #
        #reco_mf = super_concat([reco_train_mf,reco_valid_mf])
        #class_mf = super_concat([class_train_mf,class_valid_mf])
#
        #reco_df = reco_mf.to_pandas()
        #class_df = class_mf.to_pandas()
        #common_cols = [col for col in reco_df.columns if col in class_df.columns]
        #output_df = pd.merge(reco_df,class_df,on=common_cols,how='outer')
        #output_df.to_parquet(output_df_path)
        #print (f'Saved output df to {output_df_path}')


if __name__ == "__main__":
    input_mf_path = sys.argv[1]
    output_mf_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    model_path = sys.argv[4]
    results_dir = sys.argv[5] if len(sys.argv) > 5 else None

    train(input_mf_path, output_mf_path, isVal, model_path, results_dir)
