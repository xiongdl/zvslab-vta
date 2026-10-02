"""Apply validated per-occurrence snapshots through VTA's normal lowering."""

from contextlib import contextmanager

from tvm.autotvm.task.dispatcher import DispatchContext


class _OccurrenceConfigContext(DispatchContext):
    """Select captured AutoTVM configs for one actual deployment occurrence."""

    def __init__(self, entries, configs):
        super().__init__()
        if len(entries) != len(configs):
            raise ValueError("selected config count does not match captured workload count")
        self._configs = {
            (str(target), tuple(workload)): config
            for (_, workload, target, _), config in zip(entries, configs)
            if config is not None
        }

    def _query_inside(self, target, workload):
        return self._configs.get((str(target), tuple(workload)))

    def update(self, target, workload, config):
        # Bindings are immutable for the duration of one layer lowering.
        return None


@contextmanager
def occurrence_config_context(layer, selected):
    """Install selected config entities for a layer, restoring dispatch on exit."""
    if len(layer.config_spaces) != len(selected.configs):
        raise ValueError(f"schedule config count does not match occurrence {layer.occurrence}")
    context = _OccurrenceConfigContext(layer.config_spaces, selected.configs)
    with context:
        yield context


def lower_selected_deployment(module, deployment, snapshot, compiler, compiler_config):
    """Lower each outlined real Relay function under its occurrence config.

    A fresh dispatch context is used per occurrence so repeated AutoTVM
    workloads can carry different configs without leaking between layers.
    Unselected occurrences use VTA's ordinary default dispatch.
    """
    from tvm import relay

    from vta.relay import transform

    functions = transform._collect_vta_relay_functions(module)
    for function in functions:
        transform._validate_vta_function(function, compiler_config)
    if not functions:
        raise ValueError("prepared module has no VTA functions to lower")

    outlined = relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols("vta")(module)
    rows = transform._global_vta_relay_functions(outlined)
    global_handles = {function.handle.value for _, function in rows}
    nested = [
        function for function in transform._collect_vta_relay_functions(outlined)
        if function.handle.value not in global_handles
    ]
    if nested:
        raise ValueError("all nested Compiler='vta' functions must be directly outlineable")
    if len(rows) != len(deployment.layers):
        raise ValueError("prepared VTA occurrence count changed after schedule validation")
    for occurrence, ((global_var, function), layer) in enumerate(zip(rows, deployment.layers)):
        symbol = function.attrs.get_str("global_symbol")
        if symbol != layer.symbol or layer.occurrence != occurrence:
            raise ValueError("prepared VTA occurrence order changed after schedule validation")
        compiler.clear()
        selected = snapshot.selected.get(occurrence)
        try:
            if selected is None:
                primfunc = transform.lower_vta_function(function, compiler_config)
            else:
                with occurrence_config_context(layer, selected):
                    primfunc = transform.lower_vta_function(function, compiler_config)
            outlined.update_func(global_var, primfunc)
        finally:
            compiler.clear()
    return outlined
