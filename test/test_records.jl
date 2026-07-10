using ClockGradients: GradientRecord, reconstructed_enabling_times, run_recorded
using CompetingClocks: recorded_firings

const REC_MODEL = MachineRepair(5)
const REC_THETA = [0.5, 1.5]
const REC_HORIZON = 8.0
const REC_METHODS = (FirstReactionMethod(), NextReactionMethod())

testset_if("records: the Bookkeeper te reconstruction equals the recorder-stamped te on every firing and both samplers, so recorder ingestion passes the two-sided audit") do
    # The record builder never reads te from the recorder; it rebuilds it from
    # the model's enabling rule and the fired keys. Ingestion asserts the two
    # agree, so a successful GradientRecord construction IS the audit passing.
    # Exact equality (not a tolerance) because both are the same context time of
    # the same enable! event.
    for method in REC_METHODS
        rec = run_recorded(Xoshiro(20260709), REC_MODEL, REC_THETA, method; horizon=REC_HORIZON)
        firings = recorded_firings(rec)
        # A long horizon must exercise many firings, else equality is asserted
        # on a near-empty set and proves nothing.
        @test length(firings) > 20
        keys = [fr.clock for fr in firings]
        times = [fr.when for fr in firings]
        reconstructed = reconstructed_enabling_times(REC_MODEL, keys, times)
        @test reconstructed == [fr.te for fr in firings]
        # Ingestion runs the same audit internally and must not throw.
        gr = GradientRecord(REC_MODEL, rec; coupling=:redraw)
        @test length(gr) == length(firings)
        @test gr.coupling == :redraw
    end
end

testset_if("records: a corrupted enabling rule makes recorder ingestion throw the descriptive te-audit error") do
    # The audit is only meaningful if it actually fires on disagreement. A model
    # whose `enabled` rule contradicts the sampler's must be rejected loudly.
    rec = run_recorded(Xoshiro(1), REC_MODEL, REC_THETA, FirstReactionMethod(); horizon=REC_HORIZON)
    # WrongModel shares MachineRepair's clocks but mis-times enabling by claiming
    # every clock is enabled from t=0 (never cancelling), which desynchronizes te.
    @test_throws ArgumentError GradientRecord(WrongModel(REC_MODEL), rec; coupling=:redraw)
end

testset_if("records: bare-trace construction derives the retained log-uniform to within 1e-12 of the recorder-stored logu on both samplers") do
    # Identity (R) rebuilds each firing's survival log-uniform from the (key,
    # time) trace and θ0 alone. Because in v0 draw_step == enable_step it reduces
    # to the recorder's own logccdf back-calculation, so agreement is bounded by
    # the logccdf∘invlogccdf round-trip — the DerivedDraws bound of ~4e-15; we
    # assert the safer 1e-12.
    for method in REC_METHODS
        rec = run_recorded(Xoshiro(4242), REC_MODEL, REC_THETA, method; horizon=REC_HORIZON)
        firings = recorded_firings(rec)
        ingested = GradientRecord(REC_MODEL, rec; coupling=:redraw)
        keys = [fr.clock for fr in firings]
        times = [fr.when for fr in firings]
        derived = GradientRecord(REC_MODEL, REC_THETA, keys, times, REC_HORIZON; coupling=:redraw)
        @test length(derived) == length(ingested)
        # The structural back-references must be bitwise identical.
        @test derived.enable_step == ingested.enable_step
        @test derived.draw_step == ingested.draw_step
        @test maximum(abs.(derived.logu .- ingested.logu)) <= 1e-12
    end
end

testset_if("records: an unknown coupling label is rejected") do
    rec = run_recorded(Xoshiro(7), REC_MODEL, REC_THETA, FirstReactionMethod(); horizon=REC_HORIZON)
    @test_throws ArgumentError GradientRecord(REC_MODEL, rec; coupling=:bogus)
end
