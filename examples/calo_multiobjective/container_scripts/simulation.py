import json
import os
import sys
import random
import itertools

import numpy as np
import pandas as pd
from G4Calo import GeometryDescriptor, run_batch
from minipandas import MiniFrame,concat

class Simulation():
    def __init__(
            self,
            parameter_dict: dict
            ):
        self.parameter_dict = parameter_dict

        if "num_events" in parameter_dict:
            self.n_events = parameter_dict["num_events"]
        else:
            self.n_events = 100
        if "minEnergy_GeV" in parameter_dict:
            self.minEnergy_GeV = max(1e-3,parameter_dict["minEnergy_GeV"])
        else:
            self.minEnergy_GeV = 1.
        if "maxEnergy_GeV" in parameter_dict:
            self.maxEnergy_GeV = max(self.minEnergy_GeV,parameter_dict["maxEnergy_GeV"])
        else:
            self.maxEnergy_GeV = min(self.minEnergy_GeV,20.)
        if "sharedEnergy" in parameter_dict and parameter_dict['sharedEnergy']:
            self.sharedEnergy = True
        else:
            self.sharedEnergy = False
        if "exclusiveSimulation" in parameter_dict and parameter_dict['exclusiveSimulation']:
            self.exclusiveSimulation = True
        else:
            self.exclusiveSimulation = False
        self.part_numbers = {}
        for key,config in parameter_dict.items():
            if key.startswith('N:'):
                self.part_numbers[key.replace('N:','')] = np.arange(config['min_value'],config['max_value']+1)
        assert len(self.part_numbers) > 0

        self.cw = GeometryDescriptor()

        for i in range(parameter_dict['num_layers']):
            self.cw.addLayer(
                max(parameter_dict[f"thickness_absorber_{i}"], 1e-3),
                parameter_dict[f"material_absorber_{i}"],
                False,
                1,
            )
            if f'granularity:{i}' in parameter_dict.keys():
                granularity = parameter_dict[f"granularity:{i}"]
            else:
                granularity = 1
            self.cw.addLayer(
                max(parameter_dict[f"thickness_scintillator_{i}"], 1e-3),
                parameter_dict[f"material_scintillator_{i}"],
                True,
                granularity,
            )

    def run_simulation(self) -> pd.DataFrame:
        mfs = []
        N_counts = [[] for _ in range(len(self.part_numbers))]
        names = list(self.part_numbers.keys())
        for i,Ns in enumerate(itertools.product(*self.part_numbers.values())):
            if sum(Ns) == 0:
                continue
            if self.exclusiveSimulation:
                if sum(bool(N) for N in Ns) != 1:
                    continue
            parts = [name for name,N in zip(names,Ns) for _ in range(N)]
            print (f'Particles used in G4Calo : {parts} ({self.n_events} events)')
            mf: MiniFrame = run_batch(
                gd=self.cw,
                nEvents=self.n_events,
                particleSpec=parts,
                minEnergy_GeV=self.minEnergy_GeV*sum(Ns) if not self.sharedEnergy else self.minEnergy_GeV,
                maxEnergy_GeV=self.maxEnergy_GeV*sum(Ns) if not self.sharedEnergy else self.maxEnergy_GeV,
                no_mp=True,
                manual_seed=42,
            )
            mfs.append(mf)
            for j in range(len(Ns)):
                N_counts[j].append(np.full(len(mf),Ns[j],dtype='float32'))
        mf = concat(mfs, axis=0, ignore_index=True)
        for j in range(len(N_counts)):
            assert names[j] not in mf
            mf._data[f'N:{names[j]}'] = np.concatenate(N_counts[j],axis=0)
            mf._data[f'contains:{names[j]}'] = mf[f'N:{names[j]}'] > 0
        return mf


if __name__ == "__main__":
    parameter_dict_file_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(parameter_dict_file_path, "r") as file:
        parameter_dict = json.load(file)

    generator = Simulation(parameter_dict)
    mf = generator.run_simulation()

    mf.to_pickle(output_path)
    #os.system("rm -f ./*.pkl")