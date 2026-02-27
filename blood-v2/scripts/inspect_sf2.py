#!/usr/bin/env python3
"""Inspect Sample Factory 2 Learner class for extension points."""

import sample_factory
import sample_factory.algo.learning.learner as learner_module
import inspect

print(f"Sample Factory version: {sample_factory.__version__}")
print("\n=== Learner class methods ===")

for name, method in inspect.getmembers(learner_module.Learner, predicate=inspect.ismethod):
    if not name.startswith('_'):
        sig = inspect.signature(method)
        print(f"{name}{sig}")

print("\n=== Learner._calculate_losses signature ===")
calc_losses = learner_module.Learner._calculate_losses
print(inspect.signature(calc_losses))
print("\nSource:")
print(inspect.getsource(calc_losses)[:500])