"""Analytic ground-truth tests for the RCWA implementation in layers.py.

Purpose
-------
This suite exists to make the planned repairs to `layers.py` safe.  Every check
below compares the solver against something that is known exactly -- a Fresnel
/ Airy coefficient for a single film, or the energy balance A = 1 - sum(T_m) -
sum(R_m) -- rather than against a previously recorded number.

Tests that currently fail are marked `xfail(strict=True)` and carry the id of
the corresponding review finding.  Strict xfail means that once a fix lands the
test turns into an XPASS *failure*, which forces the marker to be removed.  So:

    green suite today  ==  no regression
    XPASS after a fix  ==  remove the marker, the bug is gone

Findings pinned here
--------------------
C1  TE polarisation is computed as TM for every `plate=True` layer.  FIXED.
C2  GratingLayer2's core permittivity eps2 never reaches the loss budget. FIXED.
C3  Per-layer `fill` ignored in post-processing.  FIXED.
H1  eps map leaves eps = 1 on the bottom face of the structure.  FIXED.
    The tests for all four are now live regression guards, not xfails.
C4  spectrT/spectrR held raw |t|^2 rather than diffraction efficiencies.  FIXED.
    spectrA keeps its zero-order meaning; spectrA_full is the true absorption.
C5  `eps_inp != 1` violates energy conservation.  STILL OPEN.

Running it
----------
Either of these works, from any directory:

    pytest test_rcwa.py -q          # normal use
    python test_rcwa.py             # same thing, for the IDE run button
    python test_rcwa.py -k fresnel -v     # extra args pass through to pytest

`python test_rcwa.py` matters because a pytest module executed as a plain
script would otherwise just define its functions and exit silently with no
output at all.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `import layers` work regardless of where this file sits (repo root or a
# tests/ subdirectory) and of the current working directory.  Walk up from this
# file until we find the directory that holds layers.py.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / 'layers.py').is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:                                                    # pragma: no cover
    raise RuntimeError('layers.py not found in any parent of ' + str(_HERE))

from layers import GratingLayer, GratingLayer2, GratingStructure, c_light  # noqa: E402

# layers.py still calls scipy.integrate.trapz/simps and assigns 1-element
# arrays to float slots; both are deprecated and both are tracked as finding
# M14.  Silence them here so a real warning stays visible.
pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')

# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------

LAM_NM = 800.0
LAM_CM = LAM_NM * 1e-7
FREQ = c_light / LAM_CM


def fresnel_film(eps_inp, eps_film, eps_out, d_cm, lam_cm, theta_deg, pol):
    """Exact R and T of a single film between two half-spaces.

    Returns (R, T) as power coefficients referred to the incident flux.
    """
    th = np.radians(theta_deg)
    kx = np.sqrt(complex(eps_inp)) * np.sin(th)
    kz_i = np.sqrt(complex(eps_inp) - kx**2)
    kz_f = np.sqrt(complex(eps_film) - kx**2)
    kz_o = np.sqrt(complex(eps_out) - kx**2)
    if pol == 'TM':
        a, b, c = kz_i / eps_inp, kz_f / eps_film, kz_o / eps_out
    else:
        a, b, c = kz_i, kz_f, kz_o
    r01, r12 = (a - b) / (a + b), (b - c) / (b + c)
    t01, t12 = 2 * a / (a + b), 2 * b / (b + c)
    ph = np.exp(2j * np.pi / lam_cm * kz_f * d_cm)
    r = (r01 + r12 * ph**2) / (1 + r01 * r12 * ph**2)
    t = t01 * t12 * ph / (1 + r01 * r12 * ph**2)
    return float(np.abs(r) ** 2), float(np.real(c / a) * np.abs(t) ** 2)


def plate_structure(eps_film, d_nm, eps_inp=1.0, eps_out=1.0):
    return GratingStructure([GratingLayer.PlateLayer(eps=eps_film, depth=d_nm)],
                            eps_inp=eps_inp, eps_out=eps_out)


def uniform_grating_structure(eps_film, d_nm, eps_inp=1.0, eps_out=1.0):
    """The same homogeneous film, but routed through the general (non-plate)
    eigen solver: eps0 == eps1 makes every Fourier coefficient vanish except
    m = 0, so the layer is exactly homogeneous."""
    layer = GratingLayer(eps0=eps_film, eps1=eps_film, period=1.0, fill=0.5,
                         depth=d_nm, plate=False)
    return GratingStructure([layer], eps_inp=eps_inp, eps_out=eps_out)


def absorption_from_orders(structure):
    """A = 1 - sum(T_m) - sum(R_m) over all propagating diffraction orders."""
    return 1.0 - structure.spectrTfull[0].sum() - structure.spectrRfull[0].sum()


# ---------------------------------------------------------------------------
# 1. a single film must reproduce Fresnel
# ---------------------------------------------------------------------------

EPS_FILM = 4.0
D_NM = 300.0


@pytest.mark.parametrize('pol,theta', [
    ('TM', 0.0),
    ('TM', 40.0),
    ('TE', 0.0),   # TE and TM coincide at normal incidence
    ('TE', 40.0),  # C1 fixed: the plate branch now picks the TE admittance
])
def test_plate_matches_fresnel(pol, theta):
    st = plate_structure(EPS_FILM, D_NM)
    T, R = st.TR_full(FREQ, 0, Theta=theta, polar=pol)
    R_ref, T_ref = fresnel_film(1.0, EPS_FILM, 1.0, D_NM * 1e-7, LAM_CM, theta, pol)
    assert T[0] == pytest.approx(T_ref, abs=1e-10)
    assert R[0] == pytest.approx(R_ref, abs=1e-10)


@pytest.mark.parametrize('pol,theta', [
    ('TM', 0.0), ('TM', 40.0), ('TE', 0.0), ('TE', 40.0),
])
def test_uniform_grating_layer_matches_fresnel(pol, theta):
    """Same film through the general eigen path -- this one is correct today,
    which is what localises C1 to the plate fast path."""
    st = uniform_grating_structure(EPS_FILM, D_NM)
    T, R = st.TR_full(FREQ, 3, Theta=theta, polar=pol)
    R_ref, T_ref = fresnel_film(1.0, EPS_FILM, 1.0, D_NM * 1e-7, LAM_CM, theta, pol)
    assert T[3] == pytest.approx(T_ref, abs=1e-9)
    assert R[3] == pytest.approx(R_ref, abs=1e-9)


def test_plate_and_uniform_grating_agree_in_te():
    """C1 regression: the two models of one film must agree in TE as well."""
    theta = 40.0
    a = plate_structure(EPS_FILM, D_NM).TR_full(FREQ, 3, Theta=theta, polar='TE')
    b = uniform_grating_structure(EPS_FILM, D_NM).TR_full(FREQ, 3, Theta=theta, polar='TE')
    assert a[1][3] == pytest.approx(b[1][3], abs=1e-9)


@pytest.mark.parametrize('pol', ['TM', 'TE'])
@pytest.mark.parametrize('theta', [0.0, 40.0])
def test_lossless_structure_conserves_energy(pol, theta):
    """Weak but broad invariant: a lossless stack must return T + R == 1."""
    layers = [GratingLayer(eps0=1.0, eps1=6.0, period=0.35, fill=0.5, depth=150),
              GratingLayer.PlateLayer(eps=2.5, depth=200)]
    st = GratingStructure(layers, eps_inp=1.0, eps_out=1.0)
    T, R = st.TR_full(FREQ, 20, Theta=theta, polar=pol)
    assert T.sum() + R.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. partial losses must add up to the absorbed power
# ---------------------------------------------------------------------------

EPS_LOSSY = 6.0 + 0.8j
D_LOSSY_NM = 250.0


@pytest.mark.parametrize('theta', [0.0, 30.0, 60.0])
def test_partial_loss_equals_absorption_grid_inside(theta):
    """Ground truth for calculate_partial_loss.

    The z grid is kept strictly inside the structure, which avoids H1.  Under
    that condition the loss integral is exact, so this pins the normalisation
    constant 2*pi*v/(c*cos(theta)) and the whole field reconstruction.
    """
    st = plate_structure(EPS_LOSSY, D_LOSSY_NM)
    z_max = D_LOSSY_NM * (1 - 1e-9)
    st.calcTRLandPartLoss(np.array([FREQ]), 0, (5, 1601), (0.0, z_max),
                          Theta=theta, simps_rule=True, verbose=False)
    assert st.p_losses[0].sum() == pytest.approx(absorption_from_orders(st), rel=1e-6)


@pytest.mark.parametrize('theta', [0.0, 30.0])
def test_partial_loss_equals_absorption_grid_on_interface(theta):
    """Identical to the test above except that the last z sample lands exactly
    on the bottom interface -- which is what z_range=(0, total_depth) does."""
    st = plate_structure(EPS_LOSSY, D_LOSSY_NM)
    st.calcTRLandPartLoss(np.array([FREQ]), 0, (5, 1601), (0.0, D_LOSSY_NM),
                          Theta=theta, simps_rule=True, verbose=False)
    assert st.p_losses[0].sum() == pytest.approx(absorption_from_orders(st), rel=1e-3)


def test_lossless_structure_has_no_partial_loss():
    st = plate_structure(4.0, D_LOSSY_NM)
    st.calcTRLandPartLoss(np.array([FREQ]), 0, (5, 401),
                          (0.0, D_LOSSY_NM * (1 - 1e-9)),
                          Theta=0.0, simps_rule=True, verbose=False)
    assert st.p_losses[0].sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. the two spectrum conventions
#
#   field_mapping's RETURNED T, R stay the raw |t0|^2, |r0|^2 by design -- that
#   convention is deliberately preserved, and the efficiencies are published on
#   self.T_diffraction / self.R_diffraction instead.
#   The drivers must store efficiencies in spectrT/spectrR, and the true
#   absorption in spectrA_full; spectrA keeps its zero-order meaning.
# ---------------------------------------------------------------------------

def _grating_on_substrate(eps_out):
    layers = [GratingLayer(eps0=1.0, eps1=-11 + 1.6j, period=0.35, fill=0.5, depth=60),
              GratingLayer.PlateLayer(eps=15.0 + 0.15j, depth=800)]
    return GratingStructure(layers, eps_inp=1.0, eps_out=eps_out)


def test_field_mapping_return_stays_the_raw_amplitude():
    """Preserved convention: the RETURNED T, R are |t0|^2 and |r0|^2, which are
    the efficiencies only when eps_out == 1.  Kept so that existing callers are
    unaffected; the efficiencies live on the T_diffraction attributes."""
    neq = 12
    st = _grating_on_substrate(eps_out=15.0 + 0.15j)
    *_, T, R = st.field_mapping(FREQ, neq, 11, 21, 0.0, 860.0, Theta=0.0)
    assert float(np.real(T)) == pytest.approx(abs(np.squeeze(st.t)[neq])**2, rel=1e-9)
    assert float(np.real(T)) != pytest.approx(st.T_diffraction[neq], rel=1e-3)


@pytest.mark.parametrize('eps_out', [1.0, 15.0 + 0.15j])
def test_driver_stores_zero_order_efficiency(eps_out):
    """C4a: spectrT/spectrR must be efficiencies, so that spectrT equals
    spectrTfull[:, Neq] and the two drivers agree on what spectrT means."""
    neq = 12
    st = _grating_on_substrate(eps_out=eps_out)
    st.calcTRLandPartLoss(np.array([FREQ]), neq, (11, 21), (0.0, 860.0),
                          Theta=0.0, simps_rule=True, verbose=False)
    assert st.spectrT[0] == pytest.approx(st.spectrTfull[0][neq], rel=1e-12)
    assert st.spectrR[0] == pytest.approx(st.spectrRfull[0][neq], rel=1e-12)
    assert st.spectrT[0] <= 1.0 and st.spectrA[0] >= -1e-12


def test_zero_order_fix_is_a_noop_when_eps_out_is_one():
    """The narrow case the old convention relied on: for eps_inp = eps_out = 1
    the raw amplitude IS the zero-order efficiency, at any angle.  Correcting
    C4a must therefore change nothing here."""
    neq = 12
    for theta in (0.0, 35.0):
        st = _grating_on_substrate(eps_out=1.0)
        *_, T, R = st.field_mapping(FREQ, neq, 11, 21, 0.0, 860.0, Theta=theta)
        assert float(np.real(T)) == pytest.approx(st.T_diffraction[neq], rel=1e-12)
        assert float(np.real(R)) == pytest.approx(st.R_diffraction[neq], rel=1e-12)


def test_absorption_accounts_for_every_propagating_order():
    """Period 1.6 um at 800 nm -> five propagating orders on each side."""
    neq = 20
    layers = [GratingLayer(eps0=1.0, eps1=-15 + 3j, period=1.6, fill=0.5, depth=120),
              GratingLayer.PlateLayer(eps=12.0 + 0.5j, depth=400)]
    st = GratingStructure(layers, eps_inp=1.0, eps_out=1.0)
    st.calcTRLandPartLoss(np.array([FREQ]), neq, (11, 21), (0.0, 519.0),
                          Theta=0.0, simps_rule=True, verbose=False)
    assert (st.spectrTfull[0] > 0).sum() > 1, 'test needs higher orders to propagate'
    # spectrA_full is the true absorption ...
    assert st.spectrA_full[0] == pytest.approx(absorption_from_orders(st), rel=1e-9)
    # ... while spectrA deliberately keeps its zero-order meaning
    assert st.spectrA[0] == pytest.approx(1 - st.spectrT[0] - st.spectrR[0], rel=1e-12)
    assert st.spectrA[0] != pytest.approx(st.spectrA_full[0], rel=1e-3)


def test_spectrA_full_agrees_between_drivers():
    """calcTRLspectra and calcTRLandPartLoss must define spectrT and
    spectrA_full identically."""
    neq = 12
    a = _grating_on_substrate(eps_out=15.0 + 0.15j)
    a.calcTRLspectra(0, 0, vrange=np.array([FREQ]), Neq=neq, vebrose=False)
    b = _grating_on_substrate(eps_out=15.0 + 0.15j)
    b.calcTRLandPartLoss(np.array([FREQ]), neq, (11, 21), (0.0, 860.0),
                         Theta=0.0, simps_rule=True, verbose=False)
    assert a.spectrT[0] == pytest.approx(b.spectrT[0], rel=1e-9)
    assert a.spectrR[0] == pytest.approx(b.spectrR[0], rel=1e-9)
    assert a.spectrA_full[0] == pytest.approx(b.spectrA_full[0], rel=1e-9)


# ---------------------------------------------------------------------------
# 4. a non-unity incident medium
# ---------------------------------------------------------------------------

EPS_INP = 2.25


@pytest.mark.parametrize('pol,theta', [
    pytest.param('TM', 0.0), pytest.param('TM', 20.0),
    pytest.param('TE', 0.0), pytest.param('TE', 20.0),
])
@pytest.mark.xfail(strict=True, reason='C5: the TM reflected-field row in '
                                       'solver() lacks the 1/eps_inp factor '
                                       'its transmitted counterpart has, and '
                                       'R is normalised by eps_inp instead of '
                                       'sqrt(eps_inp)')
def test_non_unity_incident_medium(pol, theta):
    # non-plate layer, so that C1 cannot contaminate the TE cases
    st = uniform_grating_structure(EPS_FILM, D_NM, eps_inp=EPS_INP, eps_out=1.0)
    T, R = st.TR_full(FREQ, 3, Theta=theta, polar=pol)
    R_ref, T_ref = fresnel_film(EPS_INP, EPS_FILM, 1.0, D_NM * 1e-7, LAM_CM, theta, pol)
    assert T[3] == pytest.approx(T_ref, abs=1e-9)
    assert R[3] == pytest.approx(R_ref, abs=1e-9)


@pytest.mark.parametrize('pol', ['TM', 'TE'])
@pytest.mark.xfail(strict=True, reason='C5: energy is not conserved when '
                                       'eps_inp != 1')
def test_non_unity_incident_medium_conserves_energy(pol):
    st = uniform_grating_structure(EPS_FILM, D_NM, eps_inp=EPS_INP, eps_out=1.0)
    T, R = st.TR_full(FREQ, 3, Theta=20.0, polar=pol)
    assert T.sum() + R.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. multi-medium layers: geometry must be described consistently
# ---------------------------------------------------------------------------

def test_gratinglayer2_core_absorption_is_counted():
    """Lossless shell, absorbing core: all of the layer's loss lives in eps2."""
    core = GratingLayer2(eps0=1.0, eps1=3.1 + 0.0j, eps2=-11.0 + 1.6j,
                         period=0.6, fill=0.5, fill2=0.55, depth=50)
    st = GratingStructure([core, GratingLayer.PlateLayer(eps=15.0 + 0.15j, depth=800)],
                          eps_inp=1.0, eps_out=1.0)
    st.calcTRLandPartLoss(np.array([FREQ]), 20, (201, 801),
                          (0.0, 850.0 * (1 - 1e-9)),
                          Theta=0.0, simps_rule=True, verbose=False)
    assert st.p_losses[0][0] > 0.0, 'the absorbing core reports zero loss'
    assert st.p_losses[0].sum() == pytest.approx(absorption_from_orders(st), rel=2e-2)


def test_each_layer_uses_its_own_fill():
    fills = [0.2, 0.5, 0.8]
    eps_metal = 9.0 + 1j
    layers = [GratingLayer(eps0=1.0, eps1=eps_metal, period=1.0, fill=f, depth=100)
              for f in fills]
    st = GratingStructure(layers, eps_inp=1.0, eps_out=1.0)

    x_um = np.linspace(0, 1.0, 1001)
    z_cm = np.array([50e-7, 150e-7, 250e-7])          # mid-plane of each slice
    eps_map = st.eps_structure(*np.meshgrid(x_um, z_cm), FREQ)

    for row, f in enumerate(fills):
        drawn = float(np.mean(np.isclose(eps_map[row], eps_metal)))
        assert drawn == pytest.approx(f, abs=0.02), (
            f'slice {row}: fill {f} was drawn as {drawn:.2f}')


def test_wide_slice_strip_wraps_around_the_period_edge():
    """Each layer's strip is centred on the common Fourier origin, which the
    field routines place at Layers[0].fill/2.  A slice wider than the first one
    therefore wraps across x = 0/1, and must be drawn on BOTH sides -- the case
    a naive `x < Layer.fill` mask cannot express."""
    eps_metal = 9.0 + 1j
    layers = [GratingLayer(eps0=1.0, eps1=eps_metal, period=1.0, fill=0.2, depth=100),
              GratingLayer(eps0=1.0, eps1=eps_metal, period=1.0, fill=0.8, depth=100)]
    st = GratingStructure(layers, eps_inp=1.0, eps_out=1.0)

    x_um = np.linspace(0, 1.0, 1001)
    eps_map = st.eps_structure(*np.meshgrid(x_um, np.array([50e-7, 150e-7])), FREQ)
    wide = np.isclose(eps_map[1], eps_metal)

    # centre f0/2 = 0.1, half width 0.4  ->  [0.7, 1.0) U [0.0, 0.5)
    assert wide[np.argmin(abs(x_um - 0.05))], 'left lobe (wrapped) missing'
    assert wide[np.argmin(abs(x_um - 0.85))], 'right lobe (wrapped) missing'
    assert not wide[np.argmin(abs(x_um - 0.60))], 'gap should be background'
    assert float(np.mean(wide)) == pytest.approx(0.8, abs=0.02)


def test_partial_loss_survives_a_wrapped_strip():
    """Energy balance for the same wrapping geometry: if the loss mask lost the
    wrapped lobe the budget would come out short."""
    layers = [GratingLayer(eps0=1.0, eps1=-15 + 3j, period=0.35, fill=0.25, depth=80),
              GratingLayer(eps0=1.0, eps1=-15 + 3j, period=0.35, fill=0.75, depth=80)]
    st = GratingStructure(layers, eps_inp=1.0, eps_out=1.0)
    st.calcTRLandPartLoss(np.array([FREQ]), 24, (801, 1601), (0.0, 160.0),
                          Theta=0.0, simps_rule=True, verbose=False)
    assert st.p_losses[0].sum() == pytest.approx(absorption_from_orders(st), rel=3e-2)


# ---------------------------------------------------------------------------
# allow `python test_rcwa.py` in addition to `pytest test_rcwa.py`
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', *sys.argv[1:]]))
