"""Scan geometry: loc index k (1-based) -> map row/column -> stage coordinates.

The raster is one-directional and row-major: x is the fast axis, y the slow
axis. loc1..loc(nx) is the first row (y = y_start), each row starts at x_start.
"N steps" means N intervals, i.e. N + 1 points along that axis.
"""
from dataclasses import dataclass

import numpy as np

UNIT_TO_UM = {"mm": 1000.0, "um": 1.0, "µm": 1.0, "nm": 1e-3}


@dataclass
class ScanGrid:
    x_start: float
    x_end: float
    x_steps: int
    y_start: float
    y_end: float
    y_steps: int
    units: str = "mm"

    @property
    def nx(self):
        return self.x_steps + 1

    @property
    def ny(self):
        return self.y_steps + 1

    @property
    def npts(self):
        return self.nx * self.ny

    @property
    def x_step(self):
        return (self.x_end - self.x_start) / self.x_steps if self.x_steps else 0.0

    @property
    def y_step(self):
        return (self.y_end - self.y_start) / self.y_steps if self.y_steps else 0.0

    @property
    def to_um(self):
        return UNIT_TO_UM[self.units]

    def rowcol(self, k):
        """Map row and column (0-based) of loc index k (1-based)."""
        i = np.asarray(k) - 1
        return i // self.nx, i % self.nx

    def stage_xy(self, k):
        """Stage coordinates (in the scan's units) of loc index k."""
        r, c = self.rowcol(k)
        return self.x_start + c * self.x_step, self.y_start + r * self.y_step

    def x_axis(self):
        """Stage x of each column (scan units)."""
        return self.x_start + np.arange(self.nx) * self.x_step

    def y_axis(self):
        return self.y_start + np.arange(self.ny) * self.y_step

    def x_rel_um(self):
        """Stage x minus the scan start, in µm: the start point is at 0 (a pure
        translation, so a scan towards smaller stage values gives negative x)."""
        return np.arange(self.nx) * self.x_step * self.to_um

    def y_rel_um(self):
        return np.arange(self.ny) * self.y_step * self.to_um

    def describe(self):
        u = self.units
        return (f"x (fast): {self.x_start:g} -> {self.x_end:g} {u}, {self.x_steps} steps = {self.nx} points, "
                f"step {abs(self.x_step) * self.to_um:.4g} um\n"
                f"y (slow): {self.y_start:g} -> {self.y_end:g} {u}, {self.y_steps} steps = {self.ny} points, "
                f"step {abs(self.y_step) * self.to_um:.4g} um\n"
                f"expected {self.npts} files")
