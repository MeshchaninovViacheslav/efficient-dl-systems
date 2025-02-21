import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name: str = "model", schedule: dict = None):
        self.name_map = self._build_name_map(model, name)
        self.events = []
        self.schedule = schedule or {"wait": 1, "warmup": 1, "active": 3}  # Default schedule
        self.active = False
        self.current_step = 0
        
    def _build_name_map(self, model, name="model"):
        # Build a mapping from module objects to their display names
        name_map = {}
        
        # Iterate through all modules in the model, getting their full hierarchical names
        for full_name, module in model.named_modules():
            # For the root module, use the provided name parameter
            if full_name == "":
                full_name = name
            
            # For leaf modules just use the class name
            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            # For non-leaf modules, include the full hierarchical path and class name
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"
        
        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def __enter__(self):
        for module in self.name_map.keys():
            # Runs before the forward pass of each module
            module.register_forward_pre_hook(self._forward_pre_hook)
            # Runs after the forward pass of each module
            module.register_forward_hook(self._forward_post_hook)

            module.register_full_backward_pre_hook(self._backward_pre_hook)
            module.register_backward_hook(self._backward_post_hook)
        return self

    def __exit__(self, type, value, traceback):
        pass

    def _forward_pre_hook(self, module, inputs):
        if self.active:
            module._start_event = torch.cuda.Event(enable_timing=True)
            module._end_event = torch.cuda.Event(enable_timing=True)
            module._start_event.record()

    def _forward_post_hook(self, module, inputs, outputs):
        if self.active:
            module._end_event.record()
            torch.cuda.synchronize()  # Ensure completion
            elapsed_time = module._start_event.elapsed_time(module._end_event)
            self.events.append({"name": self.name_map[module], "type": "forward", "time": elapsed_time})

    def _backward_pre_hook(self, module, grad_output):
        if self.active:
            module._start_event = torch.cuda.Event(enable_timing=True)
            module._end_event = torch.cuda.Event(enable_timing=True)
            module._start_event.record()

    def _backward_post_hook(self, module, grad_input, grad_output):
        if self.active:
            module._end_event.record()
            torch.cuda.synchronize()  # Ensure completion
            elapsed_time = module._start_event.elapsed_time(module._end_event)
            self.events.append({"name": self.name_map[module], "type": "backward", "time": elapsed_time})

    def step(self):
        self.current_step += 1
        if self.current_step > self.schedule["wait"] + self.schedule["warmup"]:
            self.active = True

    def summary(self):
        print("Summary:")
        for event in self.events:
            print(event)

    def to_perfetto(self, path="trace.json"):
        trace_events = []
        pid = os.getpid()  # Process ID
        tid = 0  # Single-threaded

        for event in self.events:
            trace_events.append({
                "name": event["name"],
                "cat": event["type"],
                "ph": "X",
                "ts": event["time"] * 1e3,  # Convert milliseconds to microseconds
                "dur": event["time"] * 1e3,
                "pid": pid,
                "tid": tid,
            })

        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f, indent=4)

        print(f"Perfetto trace saved to {path}")

