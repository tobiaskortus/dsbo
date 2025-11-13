import os
import subprocess
from typing import List, Optional
from dsbo.simulation import SimulationContainerWrapperBase


def gate_box(parent_name: str, name: str, z_length: float, z_translation: float, material: Optional[str] = None):
    lines = [
        f"/gate/{parent_name}/daughters/name {name}",
        f"/gate/{parent_name}/daughters/insert box",
        f"/gate/{name}/geometry/setXLength 270 mm",
        f"/gate/{name}/geometry/setYLength 164 mm",
        f"/gate/{name}/geometry/setZLength {z_length/1000} mm",
        f"/gate/{name}/placement/setTranslation 0 0 {z_translation/1000} mm"]

    if material != None:
        lines += [f"/gate/{name}/setMaterial {material}"]

    lines += [f"/gate/{name}/vis/setVisible false"]
    return "\n".join(lines)


class TranslationTracker:
    def __init__(self, t0) -> None:
        self.t0 = t0

    def __call__(self, dt) -> float:
        ret_val = self.t0 + dt/2
        self.t0 = self.t0 + dt
        return ret_val


class CalorimeterLayer:
    def __init__(self, index: int, last_layer_z: float, absorber_thickness: float) -> None:
        self.index = index
        self.last_layer_z = last_layer_z * 1000
        self.abs_t = absorber_thickness * 1000
        self.tot_t = 2000 + self.abs_t
        self.z_at_abs1 = (-self.tot_t/2+self.abs_t/4) #Removed * 1000
        self.z_after_abs = (-self.tot_t/2+self.abs_t/2) #Removed *1000

        self.box_names = [[f"Glue1", f"ALPIDESub",  f"ALPIDEEpi",  f"ALPIDEFront"], 
                          [f"CapacitorAlO", f"CapacitorMetal", f"CableFDI1", f"CableFDI2", f"Glue2",
                          f"TopFDI1", f"TopFDI2", f"Glue3", f"Spacer2", f"Glue4", f"BottomFDI1",
                          f"BottomFDI2", f"Glue5"]]

        self.box_zlen = [[5, 64, 25, 11], [2.7, 1.2, 15, 10, 5, 30, 20, 5, 75, 5, 100, 20, 5]]
        self.box_mat  = [["Glue", "Silicon", "Silicon", "Silicon"], 
                         ["Al2O3", "NiCu", "Aluminium", "Kapton", "Glue", "Aluminium", "Kapton", "Glue", "Kapton", "Glue",
                          "Aluminium", "Kapton", "Glue"]]

    def construct(self):
        translation_fn = TranslationTracker(self.z_after_abs)
        geometry = [gate_box("scanner_2", f"Layer_{self.index}", self.tot_t, self.last_layer_z+self.tot_t/2, None),
                    gate_box(f"Layer_{self.index}", f"Absorber1_{self.index}", self.abs_t/2, self.z_at_abs1, "Aluminium")]

        #Geometry before air gap
        for name, zlen, mat in zip(self.box_names[0], self.box_zlen[0], self.box_mat[0]):
            geometry += [gate_box(f"Layer_{self.index}", f"{name}_{self.index}", zlen, translation_fn(zlen), mat)]

        translation_fn(1601.1)

        #Geometry after air gap
        for name, zlen, mat in zip(self.box_names[1], self.box_zlen[1], self.box_mat[1]):
            geometry += [gate_box(f"Layer_{self.index}", f"{name}_{self.index}", zlen, translation_fn(zlen), mat)]


        geometry += [gate_box(f"Layer_{self.index}", f"Absorber2_{self.index}", self.abs_t/2, translation_fn(self.abs_t/2), "Aluminium")]

        return "\n\n".join(geometry) + "\n"


class GATESimulation(SimulationContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str) -> None:
        super().__init__(workdir, container_path=container_path, container_extra_flags="--writable-tmpfs")
    
    def simulate(self, design: dict, n_particles: int = 100) -> None:
        absorber_thicknesses = design.values()
        calo_dir = os.path.join(self.workdir, "calorimeter"); os.makedirs(calo_dir, exist_ok=True)
        self.__generate_detector_file(os.path.join(self.workdir, "calorimeter"), absorber_thicknesses)
        
        subprocess.run((f"singularity run --writable-tmpfs -B {self._simulation_dir}/:/output -B {calo_dir}/:/monte-carlo-simulations/detectors/simplified/calorimeter {self.container_path} " \
                        f"--record-detector-psa --phantom water -wpt 0.01,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200 -j 8 -p {n_particles}").split(" "))
            
    
    def _generate_readout(self, num_layers):
        readout = []
        for index in range(num_layers):
            readout += [f"/gate/systems/detectorContainer/level2/attach Layer_{index}",
                        f"/gate/ALPIDEEpi_{index}/attachCrystalSD"]
        return "\n".join(readout)


    def __generate_detector_file(self, directory, thicknesses):
        assert all(t > 0 for t in thicknesses), "All absorber thicknesses must be positive."
        assert all(t <= 10 for t in thicknesses), "All absorber thicknesses must be less than or equal to 10 mm."
        
        last_layer_z = -246; calorimeter_definition = ""

        calorimeter_definition = ""
        for i, t in enumerate(thicknesses):
            calorimeter_definition += CalorimeterLayer(i, last_layer_z, t).construct() 
            last_layer_z += t + 2
        
        calorimeter_definition += self._generate_readout(len(thicknesses))

        with open(os.path.join(self.workdir, "calorimeter", "calorimeter.mac"), "w") as f:
            f.writelines(calorimeter_definition)