from importlib.metadata import version as _v
__version__ = _v(__name__)

def register():
    from .flow_stub import FluxModelStub, TrellisModelStub
    from vllm import ModelRegistry
    ModelRegistry.register_model("flux", FluxModelStub)
    ModelRegistry.register_model("trellis", TrellisModelStub)

    from .flow_executor import FlowExecutor
    from vllm import ExecutorRegistry
    ExecutorRegistry.register_executor("flow_exec", FlowExecutor)

    from .micro_sched import MicroStepScheduler
    from vllm import SchedulerRegistry
    SchedulerRegistry.register_scheduler("flow_sched", MicroStepScheduler)
