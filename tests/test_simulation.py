import pytest
from unittest.mock import patch, MagicMock

from dsbo.simulation import SimulationBase, DaskSimulationWrapper, DaskSimulationWrapper


class MockSimulation(SimulationBase):
    def __init__(self, workdir: str) -> None:
        super().__init__(workdir)

    def simulate(self, design: dict, suffix: str | None = None, trial_id: int | None = None) -> None:
        pass

# Test cases for SimulationBase 

@patch('os.makedirs')
def test_set_trial_id_when_override(mock_makedirs):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({}, trial_id=1)
    assert sim._trial_id == 1


@patch('os.makedirs')
def test_workdir_with_trial_id(mock_makedirs):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({}, trial_id=1)
    expected_workdir = "/tmp/mock_simulation/trial_1"
    assert sim.workdir == expected_workdir
    
@patch('os.makedirs')
def test_workdir_without_trial_id(mock_makedirs):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({})
    expected_workdir = "/tmp/mock_simulation"
    assert sim.workdir == expected_workdir

@patch('os.makedirs')
def test_simulation_dir_with_trial_id(mock_makedirs):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({}, trial_id=1)
    expected_simulation_dir = "/tmp/mock_simulation/trial_1/output"
    assert sim.simulation_dir == expected_simulation_dir

@patch('os.makedirs')
def test_simulation_dir_without_trial_id(mock_makedirs):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({})
    expected_simulation_dir = "/tmp/mock_simulation/output"
    assert sim.simulation_dir == expected_simulation_dir

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_create_simulation_dir(mock_makedirs, mock_exists):
    sim = MockSimulation("/tmp/mock_simulation")
    sim.simulate({}, trial_id=1)
    
    expected_path = "/tmp/mock_simulation/trial_1/output"
    mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


@patch('shutil.rmtree')
@patch('os.path.exists', return_value=True)
def test_remove_simulation_files(mock_exists, mock_rmtree):
    sim = MockSimulation("/tmp/mock_simulation")
    sim._trial_id = 1  # type: ignore
    
    sim.remove_simulation_files()
    
    expected_path = "/tmp/mock_simulation/trial_1/output"
    mock_exists.assert_called_once_with(expected_path)
    mock_rmtree.assert_called_once_with(expected_path, ignore_errors=True)


# Test cases for SimulationContainerWrapperBase

# Test cases for DaskSimulationWrapper

@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
def test_initialize_on_demand(mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, threads_per_worker=2, 
                                     merge_function=mock_merge_function, init_on_demand=True)
    
    # Verify cluster and client were not initialized during __init__ when init_on_demand=True
    mock_local_cluster.assert_not_called()
    mock_client.assert_not_called()
    assert dask_sim.init_on_demand is True
    assert not hasattr(dask_sim, 'cluster') or dask_sim.cluster is None
    assert not hasattr(dask_sim, 'client') or dask_sim.client is None


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_initialize_on_demand_called_during_simulate(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, threads_per_worker=2, 
                                     merge_function=mock_merge_function, init_on_demand=True)
    
    dask_sim.simulate({}, trial_id=1)
    
    # Verify cluster and client were initialized during simulate
    mock_local_cluster.assert_called_once_with(n_workers=2, threads_per_worker=2, 
                                                dashboard_address=None, processes=False)
    mock_client.assert_called_once()


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
def test_initialize_on_init(mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, threads_per_worker=2, 
                                     merge_function=mock_merge_function, init_on_demand=False)
    
    # Verify cluster and client were initialized during __init__ when init_on_demand=False
    mock_local_cluster.assert_called_once_with(n_workers=2, threads_per_worker=2, 
                                                dashboard_address=None, processes=False)
    mock_client.assert_called_once()
    assert dask_sim.init_on_demand is False


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_initialize_on_init_not_called_during_simulate(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, threads_per_worker=2, 
                                     merge_function=mock_merge_function, init_on_demand=False)
    
    mock_local_cluster.reset_mock()
    mock_client.reset_mock()
    
    # Call simulate and verify initialization is not called again
    dask_sim.simulate({}, trial_id=1)
    
    mock_local_cluster.assert_not_called()
    mock_client.assert_not_called()


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_simulate_submits_correct_number_of_jobs(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    n_simulations = 5
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=n_simulations, n_workers=2, 
                                     threads_per_worker=2, merge_function=mock_merge_function, init_on_demand=False)
    
    dask_sim.simulate({"param": "value"}, trial_id=1)
    
    # Verify submit was called n_simulations times
    assert mock_client_instance.submit.call_count == n_simulations
    # Verify gather was called once with the list of futures
    mock_client_instance.gather.assert_called_once()


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_simulate_calls_merge_function_with_simulation_dir(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, 
                                     threads_per_worker=2, merge_function=mock_merge_function, init_on_demand=False)
    
    dask_sim.simulate({}, trial_id=1)
    
    # Verify merge_function was called with the simulation directory
    expected_dir = "/tmp/mock_simulation/trial_1/output"
    mock_merge_function.assert_called_once_with(expected_dir)


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_simulate_closes_when_init_on_demand(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_cluster_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_local_cluster.return_value = mock_cluster_instance
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, 
                                     threads_per_worker=2, merge_function=mock_merge_function, init_on_demand=True)
    
    dask_sim.simulate({}, trial_id=1)
    
    # Verify close was called on both client and cluster
    mock_client_instance.close.assert_called_once()
    mock_cluster_instance.close.assert_called_once()


@patch('dsbo.simulation.Client')
@patch('dsbo.simulation.LocalCluster')
@patch('os.makedirs')
def test_simulate_does_not_close_when_not_init_on_demand(mock_makedirs, mock_local_cluster, mock_client):
    base_sim = MockSimulation("/tmp/mock_simulation")
    mock_merge_function = MagicMock()
    mock_client_instance = MagicMock()
    mock_cluster_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    mock_local_cluster.return_value = mock_cluster_instance
    
    dask_sim = DaskSimulationWrapper(base_simulation=base_sim, n_simulations=4, n_workers=2, 
                                     threads_per_worker=2, merge_function=mock_merge_function, init_on_demand=False)
    
    dask_sim.simulate({}, trial_id=1)
    
    # Verify client and cluster remain active
    mock_client_instance.close.assert_not_called()
    mock_cluster_instance.close.assert_not_called()