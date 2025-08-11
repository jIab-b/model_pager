class FlowRunner:
    def __init__(self, native, device):
        self.device = device
        self.pager = native.weight_pager
        self.pool = native.activation_pool

    def run_step(self, step):
        self.pager.prefetch(step.weight_handles)
        workspace = self.pool.alloc(step.workspace_bytes)
        self.pool.free(workspace)
        self.pager.evict(step.weight_handles)
