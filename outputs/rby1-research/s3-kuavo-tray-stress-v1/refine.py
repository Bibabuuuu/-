"""Predefined near-threshold acceleration refinement with the same policy."""
import stress

stress.DEST = stress.DEST / "refined"
stress.DEST.mkdir()
stress.A = (1.25, 1.375, 1.45, *stress.A)
stress.screen()
