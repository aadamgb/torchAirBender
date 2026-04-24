import torch
from torch import Tensor


class CustomGrad(torch.autograd.Function):
    """
    Custom autograd function that decouples the forward and backward passes
    of the quadrotor dynamics:

      Forward  : _step_fwd  — full motor-lag simulation 
      Backward : _step_bck  — simplified algebraic step 

    Usage:
        from surrogate_gradient import CustomGrad
        
        next_state = CustomGrad.apply(
            state, thrusts, alloc, m, J, km, a0, motor_tau, G, dt,
            _step_fwd, _step_bck,
        )
    """

    @staticmethod
    def forward(
        ctx,
        state:     Tensor,
        thrusts:   Tensor,
        alloc:     Tensor,
        m:         Tensor,
        J:         Tensor,
        km:        Tensor,
        a0:        Tensor,
        motor_tau: Tensor,
        G:         Tensor,
        dt:        float,
        step_fwd,           # callable — full forward dynamics
        step_bck,           # callable — surrogate backward dynamics
    ) -> Tensor:
        ctx.save_for_backward(state, thrusts, alloc, m, J, G)
        ctx.dt       = dt
        ctx.step_bck = step_bck

        with torch.no_grad():
            return step_fwd(
                state=state, thrusts=thrusts, alloc=alloc,
                m=m, J=J, km=km, a0=a0, motor_tau=motor_tau,
                G=G, dt=dt,
            )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        state, thrusts, alloc, m, J, G = ctx.saved_tensors
        dt       = ctx.dt
        step_bck = ctx.step_bck

        with torch.enable_grad():
            state_   = state.detach().requires_grad_(True)
            thrusts_ = thrusts.detach().requires_grad_(True)

            surrogate_out = step_bck(
                state=state_, action=thrusts_,
                alloc=alloc, m=m, J=J, G=G, dt=dt,
            )

        # _step_bck returns (B, 13); _step_fwd returns (B, 17).
        # Slice grad_output to match the surrogate output width.
        grads = torch.autograd.grad(
                surrogate_out, 
                (state_, thrusts_), 
                grad_outputs=grad_output[..., :surrogate_out.shape[-1]],
                create_graph=False
            )

        return (
            grads[0],  # state
            grads[1],  # action  (thrusts)
            None, None, None, None, None, None, None, None, None, None
        )