import pytest
from unittest.mock import patch, MagicMock

from dsbo.parameters import CompositeDesignParameter, ContinuousDesignParameter, CategoricalDesignParameter, ValueRegistry

continuous_design_parameter = ContinuousDesignParameter("param1", 0.0, 10.0)
categorical_design_parameter = CategoricalDesignParameter("param2", ["A", "B", "C"])

composite_design_parameter = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CategoricalDesignParameter("param2", ["A", "B", "C"]),
            CompositeDesignParameter(
                name="nested_composite",
                parameters=[
                    ContinuousDesignParameter("param3", -5.0, 5.0)
                ]
            )
        ]
    )

#def test_flatten_hierarchical_dict():
#    assert False

def test_iterate_flatten_single_parameter():
    flattened = list(continuous_design_parameter.iterate_flatten())

    assert len(flattened) == 1
    assert flattened[0] == continuous_design_parameter

def test_iterate_flatten_composite_parameter():
    flattened = list(composite_design_parameter.iterate_flatten())

    assert len(flattened) == 3
    assert isinstance(flattened[0], ContinuousDesignParameter)
    assert isinstance(flattened[1], CategoricalDesignParameter)
    assert isinstance(flattened[2], ContinuousDesignParameter)


def test_num_parameters_continous():
    assert continuous_design_parameter.num_parameters() == 1

def test_num_parameters_categorical_one_hot():
    assert categorical_design_parameter.num_parameters(one_hot_categorical=True) == 3
    
def test_num_parameters_categorical_not_one_hot():
    assert categorical_design_parameter.num_parameters(one_hot_categorical=False) == 1

def test_num_parameters_composite():
    assert composite_design_parameter.num_parameters(one_hot_categorical=True) == 1 + 3 + 1
    assert composite_design_parameter.num_parameters(one_hot_categorical=False) == 1 + 1 + 1

def test_value_registry_sample_from():
    design = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CategoricalDesignParameter("param2", ["A", "B", "C"]),
        ]
    )

    def get_sampler_callback():
        def sampler(param: ContinuousDesignParameter):
            if isinstance(param, ContinuousDesignParameter):
                return 0.5 
            elif isinstance(param, CategoricalDesignParameter):
                return param.choices[0] 
            else:
                raise ValueError("Unknown parameter type")
        return sampler

    sample = ValueRegistry.sample_from(design=design, sampler_callback=get_sampler_callback())

    assert sample["param1"] == 0.5
    assert sample["param2"] == "A"

def test_value_registry_immutable():
    design = ContinuousDesignParameter("param1", 0.0, 1.0)

    def sampler(param: ContinuousDesignParameter):
        return 0.5

    sample = ValueRegistry.sample_from(design=design, sampler_callback=sampler)

    with pytest.raises(AttributeError):
        sample["param1"] = 0.8

    with pytest.raises(AttributeError):
        sample["param1"] = 0.8


def test_value_registry_getitem_invalid_key():
    design = ContinuousDesignParameter("param1", 0.0, 1.0)

    def sampler(param: ContinuousDesignParameter):
        return 0.5

    sample = ValueRegistry.sample_from(design=design, sampler_callback=sampler)

    with pytest.raises(AttributeError) as exc_info: 
        _ = sample["invalid_key"]
        assert "'ValueRegistry' object has no attribute'" in str(exc_info.value)


def test_value_registry_getitem_nested():
    design = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CompositeDesignParameter(
                name="nested_composite",
                parameters=[
                    CategoricalDesignParameter("param2", ["A", "B", "C"])
                ]
            )
        ]
    )

    def get_sampler_callback():
        def sampler(param: ContinuousDesignParameter):
            if isinstance(param, ContinuousDesignParameter):
                return 0.5 
            elif isinstance(param, CategoricalDesignParameter):
                return param.choices[0] 
            else:
                raise ValueError("Unknown parameter type")
        return sampler

    sample = ValueRegistry.sample_from(design=design, sampler_callback=get_sampler_callback())

    assert sample["param1"] == 0.5
    assert sample["param2"] == "A"


def test_value_registry_getitem_flat():
    design = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CategoricalDesignParameter("param2", ["A", "B", "C"]),
        ]
    )

    def get_sampler_callback():
        def sampler(param: ContinuousDesignParameter):
            if isinstance(param, ContinuousDesignParameter):
                return 0.5 
            elif isinstance(param, CategoricalDesignParameter):
                return param.choices[0] 
            else:
                raise ValueError("Unknown parameter type")
        return sampler

    sample = ValueRegistry.sample_from(design=design, sampler_callback=get_sampler_callback())

    assert sample["param1"] == 0.5
    assert sample["param2"] == "A"


def test_value_registry_to_design_vector():
    design = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CategoricalDesignParameter("param2", ["A", "B", "C"]),
            CompositeDesignParameter(
                name="nested_composite",
                parameters=[
                    ContinuousDesignParameter("param3", -5.0, 5.0),
                    CategoricalDesignParameter("param4", ["X", "Y"] )]
            )
        ]
    )

    def get_sampler_callback():
        def sampler(param: ContinuousDesignParameter):
            if isinstance(param, ContinuousDesignParameter):
                return 0.5 
            elif isinstance(param, CategoricalDesignParameter):
                return param.choices[1] 
            else:
                raise ValueError("Unknown parameter type")
        return sampler

    sample = ValueRegistry.sample_from(design=design, sampler_callback=get_sampler_callback())
    design_vector = sample.to_design_vector(one_hot_categorical=True)

    assert len(design_vector) == 1 + 3 + 1 + 2
    assert design_vector == [0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0]

def test_value_registry_from_design_vector():
    design = CompositeDesignParameter(
        name="composite",
        parameters=[
            ContinuousDesignParameter("param1", 0.0, 1.0),
            CategoricalDesignParameter("param2", ["A", "B", "C"]),
            CompositeDesignParameter(
                name="nested_composite",
                parameters=[
                    ContinuousDesignParameter("param3", -5.0, 5.0),
                    CategoricalDesignParameter("param4", ["X", "Y"] )]
            )
        ]
    )

    design_vector = [0.7, 0.0, 1.0, 0.0, 0.3, 0.0, 1.0]

    sample = ValueRegistry.from_design_vector(design=design, design_vector=design_vector, one_hot_categorical=True)

    assert sample["param1"] == 0.7
    assert sample["param2"] == "B"
    assert sample["param3"] == 0.3
    assert sample["param4"] == "Y"