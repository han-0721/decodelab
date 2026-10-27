import torch


def pytest_sessionstart(session):
    # Tiny matrix operations run more predictably with one CPU thread.
    torch.set_num_threads(1)
