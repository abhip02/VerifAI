"""``python -m budget_sweep`` → :func:`budget_sweep.main.main`."""

import multiprocessing as mp

from .main import main

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
