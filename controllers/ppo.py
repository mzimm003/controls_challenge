from . import BaseController
import numpy as np
import torch
from tinyphysics import FuturePlan, State

class PPOControllerConfig:
    def __init__(
            self,
            width=64,
            depth=3
            ):
        self.width = width
        self.depth = depth

class PPOControllerEvaluator(torch.nn.Module):
    def __init__(self, config:PPOControllerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lataccel_norm = torch.nn.BatchNorm1d(1)
        self.roll_norm = torch.nn.BatchNorm1d(1)
        self.a_norm = torch.nn.BatchNorm1d(1)
        self.v_norm = torch.nn.BatchNorm1d(1)
        self.norm_activation = torch.nn.Tanh()
        self.reg = torch.nn.Sequential(*
            [torch.nn.Sequential(*[
                torch.nn.Linear(201,config.width),
                torch.nn.BatchNorm1d(config.width),
                self.norm_activation
                ])]
            +[torch.nn.Sequential(*[
                torch.nn.Linear(config.width,config.width),
                torch.nn.BatchNorm1d(config.width),
                self.norm_activation
                ])
                for _ in range(config.depth)
                ]
            +[torch.nn.Linear(config.width,1)]
        )

    def forward(
            self,
            target_lataccel:float,
            current_lataccel:float,
            state:State,
            future_plan:FuturePlan):
        lataccel_inp = torch.tensor([
            target_lataccel,
            current_lataccel,
            *future_plan.lataccel], dtype=torch.float32).unsqueeze(-1)
        lataccel_inp_norm = self.lataccel_norm(lataccel_inp)
        roll_lataccel_inp = torch.tensor([
            state.roll_lataccel,
            *future_plan.roll_lataccel], dtype=torch.float32).unsqueeze(-1)
        roll_lataccel_inp_norm = self.roll_norm(roll_lataccel_inp)
        a_inp = torch.tensor([
            state.a_ego, *future_plan.a_ego
        ], dtype=torch.float32).unsqueeze(-1)
        a_inp_norm = self.a_norm(a_inp)
        v_inp = torch.tensor([
            state.v_ego, *future_plan.v_ego
        ], dtype=torch.float32).unsqueeze(-1)
        v_inp_norm = self.v_norm(v_inp)

        inp = torch.concat([
            lataccel_inp_norm,
            roll_lataccel_inp_norm,
            a_inp_norm,
            v_inp_norm]).squeeze(-1).unsqueeze(0)
        return self.reg(self.norm_activation(inp))

class Controller(BaseController):
    def __init__(self, config:PPOControllerConfig=None):
        self.config = PPOControllerConfig() if config is None else config
        self.evaluator = PPOControllerEvaluator(config=self.config)
        self.evaluator.eval()

    def update(self, target_lataccel:float, current_lataccel:float, state:State, future_plan:FuturePlan):
        """
        Args:
        target_lataccel: The target lateral acceleration.
        current_lataccel: The current lateral acceleration.
        state: The current state of the vehicle.
        future_plan: The future plan for the next N frames.
        Returns:
        The control signal to be applied to the vehicle.
        """
        if len(future_plan.a_ego) < 49:
            if not future_plan.a_ego:
                current = dict(lataccel=current_lataccel,**state._asdict())
                future_plan = type(future_plan)(**dict(
                    (k, [v])
                    for k, v
                    in current.items()
                ))
            diff = 49 - len(future_plan.a_ego)
            future_plan = type(future_plan)(*[
                x + x[-1:]*diff for x in future_plan
            ])

        return self.evaluator(
            target_lataccel=target_lataccel,
            current_lataccel=current_lataccel,
            state=state,
            future_plan=future_plan).item()

if __name__ == "__main__":
    c = Controller()