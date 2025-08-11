from vllm.executor.gpu_executor import GPUExecutor
from .flow_worker import FlowWorker

class FlowExecutor(GPUExecutor):
    uses_ray = False

    def _init_executor(self):
        self.driver_worker = FlowWorker(
            self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )

    async def _driver_execute_model_async(self, exec_req):
        return await self.driver_worker.execute_model_async(exec_req)
