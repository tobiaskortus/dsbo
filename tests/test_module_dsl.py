import pytest

from dsbo.modules.dsl import ModuleDslMixin, DependencyDict, dynamic, Dynamic

@pytest.fixture(scope="session", autouse=True)
def depdict_obj_factory():
        def depdict_obj():
            module1 = ModuleDslMixin()
            module2 = ModuleDslMixin()
            module3 = ModuleDslMixin()
            depdict = DependencyDict({module1: [module2, module3]})
            return module1, module2 , module3, depdict  
        return depdict_obj


@pytest.fixture(scope="session", autouse=True)
def depdict_obj_many_factory():
    def depdict_obj_many():
        module1 = ModuleDslMixin()
        module2 = ModuleDslMixin()
        module3 = ModuleDslMixin()
        module4 = ModuleDslMixin()
        depdict = DependencyDict({module1: [module3, module4],
                                module2: [module3, module4]})
        
        return module1, module2 , module3, module4, depdict
    return depdict_obj_many


def test_module_replicate_index():
    module = ModuleDslMixin()
    module0 = module.replicate(0)

    with pytest.raises(AttributeError):
        module.i 

    assert module0.i == 0


def test_module_replicate_shallow_copy():
    module = ModuleDslMixin()
    class SomeObject:
        def __init__(self) -> None:
            self.var1 = 1
            self.var2 = 2

    obj = SomeObject()
    module.obj = obj

    module0 = module.replicate(0)

    assert module0.obj == module.obj


def test_module_or_module_module():
    module1 = ModuleDslMixin()
    module2 = ModuleDslMixin()

    module_or = module1 | module2

    assert isinstance(module_or, DependencyDict)
    assert module_or == {module1: [], module2: []}


def test_module_or_module_depdict(depdict_obj_factory):
    module1 = ModuleDslMixin()
    module2, module3, module4, depdict = depdict_obj_factory()

    module_or = module1 | depdict
    
    assert isinstance(module_or, DependencyDict)
    assert module_or == {module1: [], 
                         module2: [module3, module4]}


def test_module_rshift_module_module():
    module1 = ModuleDslMixin()
    module2 = ModuleDslMixin()

    module_rshift = module1 >> module2

    assert isinstance(module_rshift, DependencyDict)
    assert module_rshift == {module1: [module2]}


def test_module_rshift_module_depdict_one_to_one(depdict_obj_factory):
    module1 = ModuleDslMixin()
    module2, module3, module4, depdict = depdict_obj_factory()

    module_rshift = module1 >> depdict

    assert isinstance(module_rshift, DependencyDict)
    assert module_rshift == {module1: [module2],
                             module2: [module3, module4]}
    

def test_module_rshift_module_depdict_one_to_many(depdict_obj_many_factory):
    module1 = ModuleDslMixin()
    module2, module3, module4, module5, depdict = depdict_obj_many_factory()

    depdict = DependencyDict({module2: [module4, module5],
                              module3: [module4, module5]})
    
    module_rshift = module1 >> depdict
    
    assert isinstance(module_rshift, DependencyDict)
    assert module_rshift == {module1: [module2, module3],
                             module2: [module4, module5],
                             module3: [module4, module5]}

    
def test_module_mul_int():
    module1 = ModuleDslMixin()

    module_mul = module1 * 3

    assert isinstance(module_mul, DependencyDict)
    assert len(module_mul.keys()) == 3
    
    for i, (key, val) in enumerate(module_mul.items()):
        assert key.i == i
        assert val == []




def test_depdict_or_depdict_module(depdict_obj_factory):
    module4 = ModuleDslMixin()
    module1, module2, module3, depdict = depdict_obj_factory()

    depdict_or = depdict | module4

    assert isinstance(depdict_or, DependencyDict)
    assert depdict_or == {module1: [module2, module3],
                          module4: []}


def test_depdict_or_depdict_depdict(depdict_obj_factory):
    module1, module2, module3, depdict1 = depdict_obj_factory()
    module4, module5, module6, depdict2 = depdict_obj_factory()

    depdict_or = depdict1 | depdict2

    assert isinstance(depdict_or, DependencyDict)
    assert depdict_or == {module1: [module2, module3],
                          module4: [module5, module6]}
    

def test_depdict_mul_int(depdict_obj_factory):
    _, _, _, depdict = depdict_obj_factory()

    depdict_mul = depdict * 3

    assert isinstance(depdict_mul, DependencyDict)
    assert len(depdict_mul.keys()) == 3

    for i, (key, val) in enumerate(depdict_mul.items()):
        assert key.i == i
        assert len(val) == 2
        for v in val:
            assert v.i == i


