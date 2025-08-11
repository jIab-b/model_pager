from vllm.worker.worker_base import WorkerBase, ExecuteModelRequest
from .flow_runner import FlowRunner
from .csrc_loader import load_native_libs

class FlowWorker(WorkerBase):
    def init_device(self):
        self.native = load_native_libs()

        self.runner = FlowRunner(self.native, self.device)

    def prepare_worker_input(self, req: ExecuteModelRequest):
        return req

    def execute_worker(self, req):
        for step in req.step_plan:
            self.runner.run_step(step)
        return None
