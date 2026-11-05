"""Area-normalised peak lineshapes, parameterised by centre and widths.

All functions take x (any unit, here eV), centre x0, area A and widths, and
return A * profile(x) with integral(profile) = 1. Fitting in energy makes the
Lorentzian/Gaussian/Voigt parameters physically meaningful.

Parameters per shape (after x0 and A):
    lorentzian : fwhm
    gaussian   : fwhm
    voigt      : fwhm_g (Gaussian FWHM), fwhm_l (Lorentzian FWHM)
"""
import numpy as np

LN2 = np.log(2.0)
SHAPES = ("lorentzian", "gaussian", "voigt")


def lorentzian(x, x0, area, fwhm):
    g = fwhm / 2.0
    return area * g / np.pi / ((x - x0) ** 2 + g ** 2)


def gaussian(x, x0, area, fwhm):
    s = fwhm / (2.0 * np.sqrt(2.0 * LN2))
    return area * np.exp(-0.5 * ((x - x0) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def weideman_coeffs(N=16):
    """Coefficients of Weideman's (1994) rational approximation of the Faddeeva
    function w(z). N=16: relative error ~1e-7; N=32: ~1e-14."""
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    t = L * np.tan(k * np.pi / M2)
    f = np.concatenate([[0.0], np.exp(-t ** 2) * (L ** 2 + t ** 2)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    return float(L), np.flipud(a[1:N + 1])


_WL, _WA = weideman_coeffs(16)


def faddeeva(z, L=_WL, a=_WA):
    """w(z) for Im z >= 0 (Weideman 1994). ~5x faster than scipy.special.wofz."""
    iz = 1j * z
    Z = (L + iz) / (L - iz)
    p = np.polyval(a, Z)
    return 2 * p / (L - iz) ** 2 + (1 / np.sqrt(np.pi)) / (L - iz)


def voigt(x, x0, area, fwhm_g, fwhm_l):
    s = fwhm_g / (2.0 * np.sqrt(2.0 * LN2))
    r = s * np.sqrt(2.0)
    return area * faddeeva((x - x0 + 0.5j * fwhm_l) / r).real / (s * np.sqrt(2.0 * np.pi))


def voigt_fwhm(fwhm_g, fwhm_l):
    """Olivero & Longbothum (1977) approximation, accurate to ~0.02 %."""
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)


FUNCS = {"lorentzian": lorentzian, "gaussian": gaussian, "voigt": voigt}
NWIDTH = {"lorentzian": 1, "gaussian": 1, "voigt": 2}


def nparams(shape):
    return 2 + NWIDTH[shape]


def evaluate(shape, x, p):
    return FUNCS[shape](x, *p)


def fwhm(shape, p):
    """Total FWHM from a parameter vector (x0, area, widths...)."""
    if shape == "voigt":
        return voigt_fwhm(p[2], p[3])
    return p[2]


def height(shape, p):
    """Peak height (value at the centre)."""
    return evaluate(shape, np.array([p[0]]), p)[0]
