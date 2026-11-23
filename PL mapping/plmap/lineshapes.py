"""Area-normalised peak lineshapes, parameterised by centre and widths.

All functions take x (any unit, here eV), centre x0, area A and widths, and
return A * profile(x) with integral(profile) = 1. Fitting in energy makes the
Lorentzian/Gaussian/Voigt parameters physically meaningful.

Parameters per shape (after x0 and A):
    lorentzian : fwhm
    gaussian   : fwhm
    voigt      : fwhm_g (Gaussian FWHM), fwhm_l (Lorentzian FWHM)
    trion      : fwhm_g (Gaussian FWHM), tail (1/e length of the low-energy tail)

`trion` is the recoil lineshape of a charged exciton: a Gaussian convolved with a
one-sided exponential on the LOW-energy side (an exponentially modified Gaussian).
A trion with centre-of-mass momentum K emits a photon of energy
E = E_T(0) - (M_X / m_e) * hbar^2 K^2 / (2 M_T), because the leftover electron
carries the recoil. A thermal (Boltzmann) K distribution therefore gives an
exponential tail exp(-(E_T(0) - E) / tail) with tail = (M_X / m_e) * k_B * T
(about 2 k_B T in WSe2). x0 is E_T(0), the zero-momentum trion energy (the
high-energy edge), not the energy of maximum intensity.
Refs: Esser, Runge & Zimmermann, PRB 62, 8232 (2000); Ross et al., Nat. Commun.
4, 1474 (2013); Plechinger et al., pss RRL 9, 457 (2015); Christopher et al.,
Sci. Rep. 7, 14062 (2017).
"""
import numpy as np

LN2 = np.log(2.0)
SHAPES = ("lorentzian", "gaussian", "voigt", "trion")


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


def erfcx(y):
    """Scaled complementary error function exp(y^2) erfc(y) for y >= 0, as w(iy)."""
    return faddeeva(1j * np.asarray(y, float)).real


def trion(x, x0, area, fwhm_g, tail):
    """Area-normalised Gaussian (FWHM fwhm_g) convolved with exp(-(x0 - x) / tail) for x < x0.
    Written with erfcx in both branches so it never overflows; tail -> 0 gives a Gaussian."""
    s = fwhm_g / (2.0 * np.sqrt(2.0 * LN2))
    u = x0 - np.asarray(x, float)                 # > 0 on the low-energy side
    z = (s / tail - u / s) / np.sqrt(2.0)
    g = np.exp(-0.5 * (u / s) ** 2)
    pos = z >= 0
    out = np.empty_like(u)
    out[pos] = g[pos] * erfcx(z[pos])
    neg = ~pos
    out[neg] = 2.0 * np.exp(0.5 * (s / tail) ** 2 - u[neg] / tail) - g[neg] * erfcx(-z[neg])
    return area * out / (2.0 * tail)


FUNCS = {"lorentzian": lorentzian, "gaussian": gaussian, "voigt": voigt, "trion": trion}
NWIDTH = {"lorentzian": 1, "gaussian": 1, "voigt": 2, "trion": 2}


def nparams(shape):
    return 2 + NWIDTH[shape]


def evaluate(shape, x, p):
    return FUNCS[shape](x, *p)


def _trion_grid(p, n=4001):
    """Fine grid covering a trion peak: 4 Gaussian FWHM above x0, tail + FWHM below."""
    x0, w, t = p[0], p[2], p[3]
    return np.linspace(x0 - 3 * w - 12 * t, x0 + 3 * w, n)


def fwhm(shape, p):
    """Total FWHM from a parameter vector (x0, area, widths...)."""
    if shape == "voigt":
        return voigt_fwhm(p[2], p[3])
    if shape == "trion":   # numerical: the profile is asymmetric
        x = _trion_grid(p)
        y = trion(x, p[0], 1.0, p[2], p[3])
        h = 0.5 * y.max()
        above = np.flatnonzero(y >= h)
        a, b = above[0], above[-1]                  # interpolate both half-maximum crossings
        xl = x[a - 1] + (h - y[a - 1]) * (x[a] - x[a - 1]) / (y[a] - y[a - 1])
        xr = x[b] + (y[b] - h) * (x[b + 1] - x[b]) / (y[b] - y[b + 1])
        return float(xr - xl)
    return p[2]


def peak_position(shape, p):
    """Energy of maximum intensity (= x0 except for the asymmetric trion)."""
    if shape == "trion":
        x = _trion_grid(p)
        return float(x[np.argmax(trion(x, p[0], 1.0, p[2], p[3]))])
    return p[0]


def height(shape, p):
    """Peak height (value at the maximum)."""
    return evaluate(shape, np.array([peak_position(shape, p)]), p)[0]
