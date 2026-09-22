package solver

import (
	"math"
	"testing"

	"richards-service/internal/constitutive"
)

// testMaterial is the preset sand-like material shared by solver tests.
func testMaterial() constitutive.Params {
	return constitutive.Params{Alpha: 6.0, N: 2.0, ThetaR: 0.05, ThetaS: 0.40, Ks: 5e-5}
}

func testGrid(nz int) Grid { return NewGrid(nz, 1.0) }

func uniformThetaSolver(t *testing.T, th0 float64, topKind string, pond float64,
	bottom string, opts Options) *Solver {
	t.Helper()
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, topKind, pond, bottom,
		repeat(th0, g.NZ), opts)
	if err != nil {
		t.Fatalf("build solver: %v", err)
	}
	return s
}

func repeat(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// frontDepth returns the deepest layer whose water content exceeds
// theta0 + frac*(thetaS-theta0); -1 if none.
func frontDepth(theta []float64, th0, thS, frac float64, g Grid) float64 {
	thr := th0 + frac*(thS-th0)
	depth := -1.0
	for i, th := range theta {
		if th > thr {
			depth = g.Z[i]
		}
	}
	return depth
}

func pondingSolver(t *testing.T, pond float64, opts Options) *Solver {
	t.Helper()
	return uniformThetaSolver(t, 0.15, TopPondedHead, pond, BottomFreeDrainage, opts)
}

func TestSingleStepMassClosure(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	res, err := s.Step(30)
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	// Storage change must equal net boundary flux depth exactly.
	want := res.StorageAfter - res.StorageBefore
	got := (res.TopFlux - res.BottomFlux) * res.Dt
	if math.Abs(want-got) > 1e-10 {
		t.Fatalf("single-step closure: dS=%v netFlux=%v diff=%v", want, got, want-got)
	}
	if math.Abs(res.MassBalanceResidual) > 1e-10 {
		t.Fatalf("reported closure residual %v", res.MassBalanceResidual)
	}
}

func TestAdaptiveMarchMassClosureEveryInterval(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k := range steps {
		if math.Abs(steps[k].MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure residual %v", k, steps[k].MassBalanceResidual)
		}
	}
	// Global closure over the whole run.
	tot := s.Storage() - steps[0].StorageBefore - (s.CumTop - s.CumBot)
	if math.Abs(tot) > 1e-9 {
		t.Fatalf("global closure residual %v", tot)
	}
}

func TestWettingFrontMovesDownAndStorageRises(t *testing.T) {
	s := pondingSolver(t, 0.02, DefaultOptions())
	steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	f1 := frontDepth(steps[50].ThetaAfter, 0.15, 0.40, 0.3, s.Grid)
	f2 := frontDepth(steps[179].ThetaAfter, 0.15, 0.40, 0.3, s.Grid)
	if f1 < 0 || f2 <= f1 {
		t.Fatalf("front did not advance: f(1530s)=%v f(5400s)=%v", f1, f2)
	}
	if steps[179].StorageAfter <= steps[0].StorageBefore {
		t.Fatal("storage did not rise")
	}
}

func TestLargerPondedHeadDeeperAndWetter(t *testing.T) {
	run := func(pond float64) (float64, float64) {
		s := pondingSolver(t, pond, DefaultOptions())
		steps, err := s.MarchAdaptive(30, 90, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march pond=%g: %v", pond, err)
		}
		last := steps[len(steps)-1]
		return frontDepth(last.ThetaAfter, 0.15, 0.40, 0.3, s.Grid), last.StorageAfter
	}
	fLo, sLo := run(0.005)
	fHi, sHi := run(0.20)
	if !(fHi > fLo+0.01) {
		t.Fatalf("front depth: low pond %v vs high pond %v", fLo, fHi)
	}
	if !(sHi > sLo+0.005) {
		t.Fatalf("storage: low pond %v vs high pond %v", sLo, sHi)
	}
}

func TestLowerKsSlowsFront(t *testing.T) {
	run := func(ks float64) float64 {
		p := testMaterial()
		p.Ks = ks
		g := testGrid(50)
		s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
			repeat(0.15, g.NZ), DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		steps, err := s.MarchAdaptive(30, 180, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march ks=%g: %v", ks, err)
		}
		last := steps[len(steps)-1]
		return frontDepth(last.ThetaAfter, 0.15, 0.40, 0.3, g)
	}
	fast := run(5e-5)
	slow := run(5e-6)
	if fast < 0.3 {
		t.Fatalf("baseline front unexpectedly shallow: %v", fast)
	}
	if slow > fast*0.5 {
		t.Fatalf("Ks/10 front %v not clearly slower than %v", slow, fast)
	}
}

func TestHydrostaticZeroFluxIsMotionless(t *testing.T) {
	p := testMaterial()
	g := testGrid(40)
	heads := make([]float64, g.NZ)
	for i := range heads {
		heads[i] = g.Z[i] - 1.5 // water table well below the column
	}
	s, err := NewSolver(p, g, TopZeroFlux, 0, BottomZeroFlux, heads, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	before := append([]float64(nil), s.Theta...)
	steps, err := s.MarchAdaptive(300, 48, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for i := range before {
		if math.Abs(s.Theta[i]-before[i]) > 1e-12 {
			t.Fatalf("layer %d drifted: %.12f -> %.12f", i, before[i], s.Theta[i])
		}
	}
	if math.Abs(s.CumTop)+math.Abs(s.CumBot) > 0 {
		t.Fatalf("zero-flux boundaries produced flux: %g %g", s.CumTop, s.CumBot)
	}
	if math.Abs(steps[len(steps)-1].MassBalanceResidual) > 1e-12 {
		t.Fatal("hydrostatic closure not zero")
	}
}

func TestWaterContentAlwaysBounded(t *testing.T) {
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.05, BottomFreeDrainage,
		repeat(0.10, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	steps, err := s.MarchAdaptive(30, 240, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	for k, st := range steps {
		for i, th := range st.ThetaAfter {
			if th < p.ThetaR-1e-12 || th > p.ThetaS+1e-12 {
				t.Fatalf("interval %d layer %d theta=%g out of [%g,%g]",
					k, i, th, p.ThetaR, p.ThetaS)
			}
		}
	}
}

func TestNonConvergenceReportedNotClipped(t *testing.T) {
	opts := DefaultOptions()
	opts.MaxIterations = 2
	s := pondingSolver(t, 0.05, opts)
	_, err := s.Step(50000.0) // deliberately brutal step with tiny iteration cap
	if err == nil {
		t.Fatal("expected convergence failure, got success")
	}
	f, ok := err.(*Failure)
	if !ok || f.Kind != FailNonConvergence {
		t.Fatalf("expected NON_CONVERGENCE failure, got %v", err)
	}
}

func TestOutOfRangeInitialRejected(t *testing.T) {
	p := testMaterial()
	g := testGrid(10)
	_, err := NewSolverFromTheta(p, g, TopZeroFlux, 0, BottomZeroFlux,
		repeat(p.ThetaS+0.01, g.NZ), DefaultOptions())
	if err == nil {
		t.Fatal("expected out-of-range initial content error")
	}
}

func TestHarmonicVsArithmeticMeanDistinguishable(t *testing.T) {
	build := func(m interblockMean) []float64 {
		s := pondingSolver(t, 0.05, DefaultOptions())
		s.mean = m
		steps, err := s.MarchAdaptive(30, 120, DefaultAdaptiveConfig())
		if err != nil {
			t.Fatalf("march mean=%v: %v", m, err)
		}
		return append([]float64(nil), steps[len(steps)-1].ThetaAfter...)
	}
	harm := build(HarmonicMean)
	arith := build(ArithmeticMean)
	maxDiff := 0.0
	for i := range harm {
		if d := math.Abs(harm[i] - arith[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff < 1e-3 {
		t.Fatalf("harmonic and arithmetic interblock K indistinguishable, max diff %v", maxDiff)
	}
	// Arithmetic mean overstates conductivity across the orders-of-magnitude
	// jump at the front, so it must predict a deeper wetting front.
	g := testGrid(50)
	fh := frontDepth(harm, 0.15, 0.40, 0.3, g)
	fa := frontDepth(arith, 0.15, 0.40, 0.3, g)
	if !(fa > fh) {
		t.Fatalf("arithmetic front %v should exceed harmonic front %v", fa, fh)
	}
}

func TestHarmonicMeanDirectlySmallerThanArithmetic(t *testing.T) {
	k1, k2 := 1e-8, 5e-5
	h := HarmonicMean.mean(k1, k2)
	a := ArithmeticMean.mean(k1, k2)
	if !(h < a) {
		t.Fatalf("harmonic %v should be strictly below arithmetic %v", h, a)
	}
	if got := HarmonicMean.mean(0, 1); got != 0 {
		t.Fatalf("harmonic mean with zero K should be 0, got %v", got)
	}
}

// TestPondedTopConductanceIsHarmonic locks the surface-to-cell-0 conductance
// of a ponded boundary to the harmonic mean of the ponded-surface Ks and the
// top-cell conductivity. The service once implemented the documented harmonic
// mean as an arithmetic mean here (while the interior faces were harmonic):
// with a nearly dry top cell the arithmetic mean stays near Ks/2 and lets
// infiltration bypass the low-conductivity surface layer.
func TestPondedTopConductanceIsHarmonic(t *testing.T) {
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
		repeat(0.10, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}

	k0 := p.ConductivitySe(thToSe(0.10, p)) // ≈ 2e-9, ~Ks/25000
	ks := p.Ks
	wantHarm := 2.0 * HarmonicMean.mean(ks, k0) / g.Dz
	wantArith := 2.0 * ArithmeticMean.mean(ks, k0) / g.Dz
	got := s.topConductance(k0)
	if math.Abs(got-wantHarm) > 1e-15*wantHarm {
		t.Fatalf("top conductance %v must be harmonic %v", got, wantHarm)
	}
	// Regression lock: the old arithmetic-mean implementation was ~1250x
	// larger at this dryness; require the throttle to stay at that level.
	if !(got < wantArith/500.0) {
		t.Fatalf("dry top layer not throttling: harmonic %v arithmetic %v", got, wantArith)
	}
	// K0 << Ks limit: harmonic(Ks,K0) -> 2*K0, so the conductance tends to
	// 4*K0/dz — the dry-cell half-distance bottleneck dominates completely.
	if r := got / (4.0 * k0 / g.Dz); math.Abs(r-1.0) > 1e-3 {
		t.Fatalf("dry-limit conductance %v should approach 4*K0/dz (ratio %v)", got, r)
	}
	// K0 = Ks limit: the two means coincide at 2*Ks/dz.
	wantSat := 2.0 * ks / g.Dz
	if got := s.topConductance(ks); math.Abs(got-wantSat) > 1e-12*wantSat {
		t.Fatalf("equal-conductivity top conductance %v want %v", got, wantSat)
	}
	// A completely dry surface cell has zero conductance (no bypass).
	if got := s.topConductance(0); got != 0 {
		t.Fatalf("zero top-cell K should give zero conductance, got %v", got)
	}
}

// TestPondedTopConductanceDerivative finite-difference checks the Newton
// Jacobian term d(gamma0)/dK0 for the harmonic surface conductance.
func TestPondedTopConductanceDerivative(t *testing.T) {
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
		repeat(0.10, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	for _, k0 := range []float64{1e-10, 2e-9, 2.2e-5, p.Ks} {
		eps := 1e-6 * k0
		fd := (s.topConductance(k0+eps) - s.topConductance(k0-eps)) / (2 * eps)
		an := s.dTopConductanceDk0(k0)
		if rel := math.Abs(fd-an) / math.Max(math.Abs(fd), 1e-30); rel > 1e-7 {
			t.Fatalf("k0=%v analytic dConductance/dK0=%v finite-diff=%v rel=%v", k0, an, fd, rel)
		}
	}
}

// drySandColumn reproduces the reported scenario: a 1 m, 50-layer sand column
// initially far below saturation under a 2 cm ponded head, free drainage at
// the bottom. alpha=6, n=2, thetaR=0.05, thetaS=0.40, Ks=5e-5.
func drySandColumn(t *testing.T, theta0 float64) *Solver {
	t.Helper()
	p := testMaterial()
	g := testGrid(50)
	s, err := NewSolverFromTheta(p, g, TopPondedHead, 0.02, BottomFreeDrainage,
		repeat(theta0, g.NZ), DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// TestDryTopLayerThrottlesFirstStepFlux locks the end-to-end first-step
// behaviour of the ponded boundary. With theta0=0.10 the top-cell K is about
// 2e-9 (Ks/25000): the first 30 s step must admit far less water than an
// initially near-saturated column, and the flux must agree (to the small
// implicit wetting over one step) with a hand harmonic-mean estimate. The
// old arithmetic-mean surface conductance made the dry-column flux hundreds
// of times too large — even larger than the near-saturated run.
func TestDryTopLayerThrottlesFirstStepFlux(t *testing.T) {
	dry, err := drySandColumn(t, 0.10).Step(30)
	if err != nil {
		t.Fatalf("dry step: %v", err)
	}
	wet, err := drySandColumn(t, 0.38).Step(30)
	if err != nil {
		t.Fatalf("wet step: %v", err)
	}
	if !(dry.TopFlux < wet.TopFlux/50.0) {
		t.Fatalf("dry first-step flux %v must be throttled well below the "+
			"near-saturated %v; a dry top layer cannot pass more water",
			dry.TopFlux, wet.TopFlux)
	}

	// Independent hand calculation at the initial state:
	// q0 = gamma0*(hP - h0 + dz/2), gamma0 the harmonic surface conductance.
	p := testMaterial()
	g := testGrid(50)
	s0 := drySandColumn(t, 0.10)
	h0 := s0.H[0]
	k0i := p.Conductivity(h0)
	qHand := (2.0 * HarmonicMean.mean(p.Ks, k0i) / g.Dz) *
		(0.02 - h0 + g.Dz/2.0)
	if r := dry.TopFlux / qHand; r < 0.5 || r > 2.0 {
		t.Fatalf("first-step flux %v must match the harmonic-mean estimate %v"+
			" (ratio %v, expected ~1 given slight implicit wetting)",
			dry.TopFlux, qHand, r)
	}
}

// TestSurfaceMeanSwitchChangesOnlyPondedFlux is the direct regression lock for
// the bug class: flipping the boundary averaging from harmonic to arithmetic
// must inflate the first ponded-boundary flux by orders of magnitude for a dry
// column (the bug), while making essentially no difference for a near-saturated
// column — exactly the asymmetry reported between the two runs.
func TestSurfaceMeanSwitchChangesOnlyPondedFlux(t *testing.T) {
	firstStepFlux := func(theta0 float64, m interblockMean) float64 {
		s := drySandColumn(t, theta0)
		s.mean = m
		res, err := s.Step(30)
		if err != nil {
			t.Fatalf("step theta=%v mean=%v: %v", theta0, m, err)
		}
		return res.TopFlux
	}
	qDryH := firstStepFlux(0.10, HarmonicMean)
	qDryA := firstStepFlux(0.10, ArithmeticMean)
	qWetH := firstStepFlux(0.38, HarmonicMean)
	qWetA := firstStepFlux(0.38, ArithmeticMean)
	if qDryA < 100.0*qDryH {
		t.Fatalf("arithmetic surface mean must inflate the dry-column first "+
			"step by orders of magnitude over harmonic: %v vs %v", qDryA, qDryH)
	}
	if r := qWetA / qWetH; math.Abs(r-1.0) > 0.05 {
		t.Fatalf("near-saturated column should be insensitive to the mean "+
			"choice, ratio %v", r)
	}
}

// TestDryPondedInfiltration30min locks the full 30-minute advance (50 layers,
// 30 s reporting interval, adaptive substepping, exactly the reported job).
// Infiltration into a nearly dry sand under only 2 cm of ponded water is
// throttled by the unsaturated surface and stays in the few-millimetre /
// shallow-front regime; the old arithmetic surface conductance admitted ~19x
// more water and drove the front through 40% of the column.
func TestDryPondedInfiltration30min(t *testing.T) {
	s := drySandColumn(t, 0.10)
	steps, err := s.MarchAdaptive(30, 60, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("march: %v", err)
	}
	last := steps[len(steps)-1]
	front := frontDepth(last.ThetaAfter, 0.10, 0.40, 0.3, s.Grid)

	// Independent upper bound on the surface supply over 1800 s: even if the
	// top cell were instantly saturated, q <= Ks*(hP+dz)/ (dz/2) ... use the
	// looser ponded driving head with Ks: q ~ Ks*(1 + 2*hP/dz) = 1.7e-4 m/s,
	// i.e. 0.306 m. The unsaturated harmonic throttle keeps it orders below.
	physicalCap := 0.02 // 20 mm of infiltrated depth for this dry-sand case
	if !(s.CumTop > 0 && s.CumTop < physicalCap) {
		t.Fatalf("30 min cumulative top inflow %v outside (0,%v)", s.CumTop, physicalCap)
	}
	if front > 0.08 {
		t.Fatalf("wetting front %v m too deep for a dry column under 2 cm pond",
			front)
	}
	// Every interval must still close exactly; the fix touches flux value,
	// never the conservation bookkeeping.
	for k, st := range steps {
		if math.Abs(st.MassBalanceResidual) > 1e-9 {
			t.Fatalf("interval %d closure residual %v", k, st.MassBalanceResidual)
		}
	}
}

func TestSingleStepAndAdaptiveIntervalSameResult(t *testing.T) {
	// When no internal substep is needed, StepAdaptive over an interval must
	// give the bit-identical result of one Step of the same length.
	direct := pondingSolver(t, 0.02, DefaultOptions())
	r1, err := direct.Step(30)
	if err != nil {
		t.Fatal(err)
	}
	adapt := pondingSolver(t, 0.02, DefaultOptions())
	agg, err := adapt.StepAdaptive(30, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatal(err)
	}
	if agg.Substeps != 1 {
		t.Fatalf("expected single substep, got %d", agg.Substeps)
	}
	for i := range r1.ThetaAfter {
		if math.Abs(r1.ThetaAfter[i]-agg.ThetaAfter[i]) > 1e-13 {
			t.Fatalf("layer %d differs: direct=%v adaptive=%v",
				i, r1.ThetaAfter[i], agg.ThetaAfter[i])
		}
	}
}

func TestAdaptiveCuttingConvergesWhereFixedStepStalls(t *testing.T) {
	// A reporting interval too long for one Newton solve at a sharp front is
	// recovered by internal halving; storage accounting still closes.
	s := pondingSolver(t, 0.05, DefaultOptions())
	agg, err := s.StepAdaptive(200, DefaultAdaptiveConfig())
	if err != nil {
		t.Fatalf("adaptive step: %v", err)
	}
	if agg.Substeps <= 1 {
		t.Fatalf("expected internal substepping, got %d", agg.Substeps)
	}
	if math.Abs(agg.MassBalanceResidual) > 1e-10 {
		t.Fatalf("adaptive closure residual %v", agg.MassBalanceResidual)
	}
}
