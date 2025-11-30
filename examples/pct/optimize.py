import numpy as np 
from scipy.interpolate import PchipInterpolator
from dsbo.parameters import CompositeDesignParameter, VectorDesignParameter


class SmoothProtonCTDesignParameterization(CompositeDesignParameter):
    def __init__(self, name: str, n_anchors: int = 5, total_thickness: float = 10.0) -> None:
        self.total_thickness = total_thickness
        parameters = [VectorDesignParameter("spline_x", n_anchors, 0, 1),
                      VectorDesignParameter("spline_y", n_anchors, 0, 1)]
        super().__init__(name, parameters)

    def parameters_to_design(self):
        def geom_curve(x: np.ndarray, y: np.ndarray, T: float, N: int):
            a_x = np.cumsum(x)    
            a_y = np.cumsum(y * T)

            C_spline = PchipInterpolator(a_x, a_y)

            x = np.linspace(0, 1, N+1)
            C = C_spline(x)
            t = np.diff(C)
            return t
        
        param_dict = super().to_dict(flatten=True)
        return geom_curve(param_dict["spline_x"],
                          param_dict["spline_y"],
                          T=self.total_thickness, N=42)
        
    def to_dict(self, flatten: bool = False) -> dict:
        design = self.parameters_to_design()
        return {f"absorber_{i}": float(t) for i, t in enumerate(design)}
    

pct_design = SmoothProtonCTDesignParameterization("proton_ct", 5, 143.5)