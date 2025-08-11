from vllm.core.scheduler import Scheduler

class MicroStepScheduler(Scheduler):
    def _schedule_default(self):
        plan = super()._schedule_default()
        for seq in plan.seq_groups:
            seq.step_plan = self._build_step_plan(seq)
        return plan

    def _build_step_plan(self, seq):
        return ["clip", "t5", "flux_iter_0", "vae"]
