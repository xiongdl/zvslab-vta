"""Apply validated per-occurrence snapshots through VTA's normal lowering."""

from dispatch import config_space_context


def cycles_within_strict_ten_percent(deployed_cycles, autotvm_cycles):
    """Return whether positive integer cycle counts differ by strictly under 10%."""
    for label, value in (("deployment", deployed_cycles), ("AutoTVM", autotvm_cycles)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} cycle count must be a positive integer")
    return 10 * abs(deployed_cycles - autotvm_cycles) < autotvm_cycles


def occurrence_config_context(layer, selected):
    """Install selected config entities for a layer, restoring dispatch on exit."""
    return config_space_context(layer.config_spaces, selected.configs)


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
