import os
from os import path

import pytest
from unittest import mock
from unittest.mock import patch

from dsbo.utils.io import AbstractFilePath, AbstractPath, DefaultPaths, Workdir, get_file_suffix, get_subdir, get_workdir, with_remote_workdir


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_path(mock_makedirs, mock_exists):
    path1 = Workdir.register('test/simulation')
    path2 = Workdir.register('test/reconstruction')
    
    assert isinstance(path1, AbstractPath)
    assert isinstance(path2, AbstractPath)
    assert Workdir.get_registered_paths() == set(['test/simulation', 'test/reconstruction'])


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_path_no_calls(mock_makedirs, mock_exists):
    path1 = Workdir.register('test/simulation')
    path2 = Workdir.register('test/simulation')
    
    assert mock_makedirs.call_count == 0
    assert mock_exists.call_count == 0


@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_register_path_exists(mock_makedirs, mock_exists):
    path = Workdir.register('test/simulation')
    path()

    assert mock_makedirs.call_count == 0


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_path_no_context(mock_makedirs, mock_exists):
    path = Workdir.register('test/simulation')
    path()  # Call the path to trigger path creation    
    
    assert mock_makedirs.call_args[0] == (os.path.join(os.getcwd(),'test', 'simulation'),)


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
@patch('dsbo.utils.io.workdir._is_dry_run', return_value=True)
def test_register_path_dry_run(mock_is_dry_run, mock_makedirs, mock_exists):
    path = Workdir.register('test/simulation')
    path_val = path()  # Call the path to trigger path creation    
    
    assert path_val is None


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_path_pathname(mock_makedirs, mock_exists):
    path = Workdir.register('test/simulation')
    assert path.pathname == 'test/simulation'


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_path_context(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        path = Workdir.register('test/simulation')
        path()  # Call the path to trigger path creation
        assert mock_makedirs.call_args[0] == ('tmp/test/simulation', )


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_get_workdir_set(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        workdir = get_workdir()
        assert workdir.root == 'tmp'


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_get_workdir_none(mock_makedirs, mock_exists):
    workdir = get_workdir()
    assert workdir is None


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_get_subdir_single(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.create_subdir('subdir'):
            subdir = get_subdir()
    
    assert subdir == 'subdir'


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_get_subdir_multiple(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.create_subdir('subdir'):
            with Workdir.create_subdir('subdir'):
                with Workdir.create_subdir('subdir'):
                    subdir = get_subdir()
    
    assert subdir == 'subdir/subdir/subdir'


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_register_workdir_file(mock_makedirs, mock_exists):
    file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt")
    assert isinstance(file, AbstractFilePath)


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_file_path(mock_makedirs, mock_exists):
    file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt")
    filepath = file()
    expected = os.path.join(os.getcwd(), DefaultPaths.SIMULATION, "data.txt")
    assert filepath == expected


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_filepath_subdir(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.create_subdir('batch=0'):
            file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt")
            filepath = file()
            expected = os.path.join('tmp', "batch=0", DefaultPaths.SIMULATION, "data.txt")
            assert filepath == expected


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')  
def test_get_file_suffix(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.use_file_suffix("0"):
            suffix = get_file_suffix()
            assert suffix == "0"

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')  
def test_get_file_suffix_nested(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.use_file_suffix("0"):
            with Workdir.use_file_suffix("0"):
                suffix = get_file_suffix()
                assert suffix == "0_0"


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_filepath_suffix_unique_false(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.use_file_suffix("0"):
            file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt", unique_filename=False)
            filepath = file()
            expected = os.path.join("tmp", DefaultPaths.SIMULATION, "data.txt")
            assert filepath == expected


@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_filepath_suffix_unique_true(mock_makedirs, mock_exists):
    with Workdir.create('tmp'):
        with Workdir.use_file_suffix("0"):
            file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt", unique_filename=True)
            filepath = file()
            expected = os.path.join("tmp", DefaultPaths.SIMULATION, "data_0.txt")
            assert filepath == expected


@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
def test_workdir_create_error_if_exists(mock_makedirs, mock_exists):
    with pytest.raises(FileExistsError) as exc_info:
        with Workdir.create('tmp'):
            pass

    assert str(exc_info.value) == "Directory tmp already exists."


@patch('os.path.exists', return_value=True)
@patch('os.makedirs')
@patch('os.listdir', return_value=[])
def test_with_remote_workdir(mock_makedirs, mock_exists, mock_listdir):    

    class RemoteModuleMock:
        def __init__(self):
            pass

        @with_remote_workdir
        def remote_function(self, *args, **kwargs):
            file = Workdir.register_file(DefaultPaths.SIMULATION, "data.txt")
            filepath = file()
            expected = os.path.join("tmp", "subdir", DefaultPaths.SIMULATION, "data_0.txt")
            assert filepath == expected
    
    RemoteModuleMock().remote_function(workdir="tmp", suffix="0", subdir="subdir")



@patch('dsbo.utils.io.workdir._is_dry_run', return_value=True)
@patch('dsbo.utils.io.Workdir.load')
@patch('dsbo.utils.io.Workdir.create_subdir')
@patch('dsbo.utils.io.Workdir.use_file_suffix')
def test_with_remote_workdir_dryrun(mock_use_file_suffix, mock_create_subdir, mock_load, mock_is_dry_run):
    class RemoteModuleMock:
        @with_remote_workdir
        def remote_function(self, *args, **kwargs):
            return "ok"

    res = RemoteModuleMock().remote_function(workdir="tmp", suffix="0", subdir="subdir")
    assert res == "ok"
    assert mock_is_dry_run.called
    assert mock_load.call_count == 0
    assert mock_create_subdir.call_count == 0
    assert mock_use_file_suffix.call_count == 0


def test_abstract_path_eq():
    a = AbstractPath("somepath")
    b = AbstractPath("somepath")

    assert a == b


def test_abstract_file_path_neq():
    a = AbstractPath("somepath")
    b = AbstractPath("someotherpath")
    
    assert a != b


def test_abstract_path_neq_incorrect_type():
    a = AbstractPath("somepath")
    
    assert a != 1

def test_abstract_file_path_eq():
    a_path = AbstractPath("somepath")

    a = AbstractFilePath(a_path, "file.npy", False)
    b = AbstractFilePath(a_path, "file.npy", False)

    assert a == b

def test_abstract_file_path_neq_path():
    a_path = AbstractPath("somepath")
    b_path = AbstractPath("someotherpath")

    a = AbstractFilePath(a_path, "file.npy", False)
    b = AbstractFilePath(b_path, "file.npy", False)

    assert a != b


def test_abstract_file_path_neq_file():
    a_path = AbstractPath("somepath")
    b_path = AbstractPath("somepath")

    a = AbstractFilePath(a_path, "file.npy", False)
    b = AbstractFilePath(b_path, "file2.npy", False)

    assert a != b

def test_abstract_file_path_neq_uniue():
    a_path = AbstractPath("somepath")
    b_path = AbstractPath("somepath")

    a = AbstractFilePath(a_path, "file.npy", False)
    b = AbstractFilePath(b_path, "file.npy", True)

    assert a != b


def test_abstract_file_path_neq_incorrect_type():
     a_path = AbstractPath("somepath")
     a = AbstractFilePath(a_path, "file.npy", False)

     assert a != 1

