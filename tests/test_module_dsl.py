from types import MappingProxyType

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


def test_dynamic_module_preserves_groups():
    source, target = ModuleDslMixin(), ModuleDslMixin()
    left = source * 3

    result = left >> target * dynamic()

    assert len(result.groups) == 2
    assert result.groups[0] is left.groups[0]
    group = result.groups[1]
    assert group.template is target
    assert group.count == 3
    assert group.nodes == result.nodes - left.nodes
    assert set(result.template_to_nodes[target]) == group.nodes
    for node in left:
        assert result[node][0].i == node.i


@pytest.mark.parametrize("dynamic_outer", [False, True])
@pytest.mark.parametrize("dynamic_inner", [False, True])
def test_nested_replication_remaps_inner_groups(dynamic_outer, dynamic_inner):
    target = ModuleDslMixin()
    inner = target * 2
    if dynamic_inner:
        inner = ModuleDslMixin() * 2 >> target * dynamic()

    result = (
        ModuleDslMixin() * 3 >> inner * dynamic()
        if dynamic_outer else inner * 3
    )

    groups = [group for group in result.groups if group.template is target]
    assert len(groups) == 3
    assert all(group.count == 2 for group in groups)
    assert len(set().union(*(group.nodes for group in groups))) == 6
    assert all(group.nodes <= result.nodes for group in result.groups)
    assert all(group.nodes.isdisjoint(inner.nodes) for group in result.groups)
    assert all(group.nodes <= inner.nodes for group in inner.groups)


@pytest.mark.parametrize("immutable", [False, True])
def test_resolver_mapping_is_copied(immutable):
    source, target = ModuleDslMixin(), ModuleDslMixin()
    mapping = {None: [target]}

    class Resolver:
        def __rrshift__(self, leaves):
            return MappingProxyType(mapping) if immutable else mapping

    result = DependencyDict({source: []}) >> Resolver()

    assert result == {source: [target]}
    assert mapping == {None: [target]}

