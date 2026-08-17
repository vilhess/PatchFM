import torch
from collections.abc import MutableMapping

class _MergedState(MutableMapping):
    def __init__(self, *states):
        self._states = [s for s in states if s is not None]

    def _owner(self, key):
        for s in self._states:
            if key in s:
                return s
        return None

    def __getitem__(self, key):
        s = self._owner(key)
        if s is None:
            raise KeyError(key)
        return s[key]

    def __setitem__(self, key, value):
        s = self._owner(key) or self._states[0]
        s[key] = value

    def __delitem__(self, key):
        s = self._owner(key)
        if s is None:
            raise KeyError(key)
        del s[key]

    def __iter__(self):
        for s in self._states:
            yield from s

    def __len__(self):
        return sum(len(s) for s in self._states)


class MuonAdamW(torch.optim.Optimizer):

    def __init__(self, params, muon_lr=0.001, muon_momentum=0.95,
                 adamw_lr=0.001, adamw_weight_decay=0.01, exceptional_params=None):
        params = list(params)
        exceptional_ids = {id(p) for p in exceptional_params} if exceptional_params is not None else set()
        muon_params = [p for p in params if p.ndim == 2 and id(p) not in exceptional_ids]
        other_params = [p for p in params if p.ndim != 2 or id(p) in exceptional_ids]

        self.optim_muon = torch.optim.Muon(muon_params, lr=muon_lr, momentum=muon_momentum) if muon_params else None
        self.optim_adamw = torch.optim.AdamW(other_params, lr=adamw_lr, weight_decay=adamw_weight_decay) if other_params else None

        self.defaults = {}

    @property
    def state(self):
        return _MergedState(
            self.optim_muon.state if self.optim_muon is not None else None,
            self.optim_adamw.state if self.optim_adamw is not None else None,
        )

    def zero_grad(self, set_to_none=True):
        if self.optim_muon is not None:
            self.optim_muon.zero_grad(set_to_none=set_to_none)
        if self.optim_adamw is not None:
            self.optim_adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.optim_muon is not None:
            self.optim_muon.step()
        if self.optim_adamw is not None:
            self.optim_adamw.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.optim_muon.state_dict() if self.optim_muon is not None else None,
            "adamw": self.optim_adamw.state_dict() if self.optim_adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.optim_muon is not None and state_dict.get("muon") is not None:
            self.optim_muon.load_state_dict(state_dict["muon"])
        if self.optim_adamw is not None and state_dict.get("adamw") is not None:
            self.optim_adamw.load_state_dict(state_dict["adamw"])

    @property
    def param_groups(self):
        groups = []
        if self.optim_muon is not None:
            groups += self.optim_muon.param_groups
        if self.optim_adamw is not None:
            groups += self.optim_adamw.param_groups
        return groups
