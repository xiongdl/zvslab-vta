"""Bind AutoTVM configurations to the captured workload occurrence."""

from contextlib import contextmanager

from tvm.autotvm.task.dispatcher import DispatchContext


class _OccurrenceDispatchContext(DispatchContext):
    def __init__(self, bindings):
        super().__init__()
        self._configs = {}
        for target, workload, config in bindings:
            key = (str(target), tuple(workload))
            if key in self._configs:
                raise ValueError(f"duplicate AutoTVM configuration binding for {key[1][0]}")
            self._configs[key] = config

    def _query_inside(self, target, workload):
        return self._configs.get((str(target), tuple(workload)))

    def update(self, target, workload, config):
        # A captured occurrence's bindings stay fixed for this lowering.
        return None


@contextmanager
def config_bindings_context(bindings):
    """Install explicit (target, workload, config) bindings for one lowering."""
    context = _OccurrenceDispatchContext(bindings)
    with context:
        yield context


def config_space_context(config_spaces, configs):
    """Bind config entities using the exact order captured for one occurrence."""
    if len(config_spaces) != len(configs):
        raise ValueError("selected config count does not match captured workload count")
    bindings = [
        (target, workload, config)
        for (_, workload, target, _), config in zip(config_spaces, configs)
        if config is not None
    ]
    return config_bindings_context(bindings)
