#![forbid(unsafe_code)]

//! EWW-K3 adaptive width policy harness.
//!
//! This runner compares one deterministic adaptive policy against every frozen
//! static width on the same synthetic width-floor trace. It measures policy
//! overhead and plane traversal separately, includes transition and verification
//! costs, and can inject one deterministic verification failure to exercise
//! rollback plus retry.
//!
//! The trace is a structural control-plane workload. It is not a model/KV,
//! GPU, TTFT, TPOT, HBM/DRAM or production performance result.

use elastic_core::{ElasticWordPlaneV1, ElasticWordWidthV1};
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const WIDTHS: [u16; 6] = [64, 128, 256, 512, 1024, 2048];
const DEMAND_BITS: [u16; 16] = [
    64, 64, 128, 256, 256, 128, 64, 512, 512, 256, 128, 64, 256, 128, 64, 64,
];
const SCHEMA: &str = "kvlab.eww-k3-adaptive-policy/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    words: usize,
    repetitions: usize,
    cooldown_epochs: usize,
    max_width_bits: u16,
    rollback_epoch: Option<usize>,
    seed: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct StageNanos {
    observation: u128,
    planning: u128,
    validation: u128,
    transition: u128,
    verification: u128,
    rollback: u128,
    traversal: u128,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct AdaptiveCounters {
    transitions: usize,
    rollbacks: usize,
    retries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PolicyState {
    width_bits: u16,
    epochs_since_transition: usize,
}

fn parse_usize(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_u16(name: &str, value: Option<String>) -> Result<u16, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<u16>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_u64(name: &str, value: Option<String>) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_config_from<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let words = parse_usize("words", args.next())?;
    let repetitions = parse_usize("repetitions", args.next())?;
    let cooldown_epochs = parse_usize("cooldown_epochs", args.next())?;
    let max_width_bits = parse_u16("max_width_bits", args.next())?;
    let rollback_epoch_raw = parse_usize("rollback_epoch_or_max", args.next())?;
    let seed = parse_u64("seed", args.next())?;
    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    if words == 0 {
        return Err("words must be non-zero".to_owned());
    }
    if repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    if !WIDTHS.contains(&max_width_bits) {
        return Err("max_width_bits must be one of 64,128,256,512,1024,2048".to_owned());
    }
    let rollback_epoch = (rollback_epoch_raw != usize::MAX).then_some(rollback_epoch_raw);
    if rollback_epoch.is_some_and(|epoch| epoch >= DEMAND_BITS.len()) {
        return Err(format!(
            "rollback_epoch must be < {} or usize::MAX",
            DEMAND_BITS.len()
        ));
    }
    Ok(Config {
        words,
        repetitions,
        cooldown_epochs,
        max_width_bits,
        rollback_epoch,
        seed,
    })
}

fn parse_config() -> Result<Config, String> {
    parse_config_from(env::args().skip(1))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn base_plane(words: usize, width_bits: u16, seed: u64) -> Result<ElasticWordPlaneV1, String> {
    let width = ElasticWordWidthV1::from_bits(width_bits).map_err(|error| error.to_string())?;
    let lanes_per_word = usize::from(width.lanes());
    let lane_count = words
        .checked_mul(lanes_per_word)
        .ok_or_else(|| "lane count overflow".to_owned())?;
    let mut lanes = vec![0_u64; lane_count];
    for word in 0..words {
        let index = u64::try_from(word).map_err(|_| "word index does not fit u64")?;
        lanes[word * lanes_per_word] = mix64(seed ^ index);
    }
    ElasticWordPlaneV1::new(width, lanes).map_err(|error| error.to_string())
}

fn touch_plane(plane: &ElasticWordPlaneV1) -> u64 {
    plane
        .as_lanes()
        .iter()
        .copied()
        .fold(0_u64, |acc, value| acc.rotate_left(7) ^ value)
}

fn verify_payload(
    expected: &ElasticWordPlaneV1,
    candidate: &ElasticWordPlaneV1,
) -> Result<(), String> {
    if expected.word_count() != candidate.word_count() {
        return Err("word count drift".to_owned());
    }
    for index in 0..expected.word_count() {
        let source = expected.word(index).map_err(|error| error.to_string())?;
        let target = candidate.word(index).map_err(|error| error.to_string())?;
        if source[0] != target[0] {
            return Err(format!("lane-0 payload drift at word {index}"));
        }
        if target.iter().skip(1).any(|value| *value != 0) {
            return Err(format!(
                "reserved target lane became non-zero at word {index}"
            ));
        }
    }
    Ok(())
}

fn choose_width(
    state: PolicyState,
    observed_floor_bits: u16,
    cooldown_epochs: usize,
    max_width_bits: u16,
) -> Result<u16, String> {
    if observed_floor_bits > max_width_bits {
        return Err(format!(
            "observed floor W{observed_floor_bits} exceeds memory-budget width W{max_width_bits}"
        ));
    }

    if observed_floor_bits > state.width_bits {
        return Ok(observed_floor_bits);
    }
    if observed_floor_bits < state.width_bits && state.epochs_since_transition >= cooldown_epochs {
        return Ok(observed_floor_bits);
    }
    Ok(state.width_bits)
}

fn materialize(
    source: &ElasticWordPlaneV1,
    target_bits: u16,
) -> Result<ElasticWordPlaneV1, String> {
    let target = ElasticWordWidthV1::from_bits(target_bits).map_err(|error| error.to_string())?;
    source
        .reference_repack_zero_extended(target)
        .map_err(|error| error.to_string())
}

fn add_duration(target: &mut u128, start: Instant) {
    *target = target.saturating_add(start.elapsed().as_nanos());
}

fn run_adaptive_once(config: Config) -> Result<(u128, StageNanos, AdaptiveCounters, u64), String> {
    let sequence_start = Instant::now();
    let mut stages = StageNanos::default();
    let mut counters = AdaptiveCounters::default();
    let mut plane = base_plane(config.words, 64, config.seed)?;
    let mut state = PolicyState {
        width_bits: 64,
        epochs_since_transition: config.cooldown_epochs,
    };
    let mut checksum = 0_u64;

    for (epoch, floor_bits) in DEMAND_BITS.into_iter().enumerate() {
        let observation_start = Instant::now();
        let observed_floor = black_box(floor_bits);
        add_duration(&mut stages.observation, observation_start);

        let planning_start = Instant::now();
        let target_bits = choose_width(
            state,
            observed_floor,
            config.cooldown_epochs,
            config.max_width_bits,
        )?;
        add_duration(&mut stages.planning, planning_start);

        let validation_start = Instant::now();
        let target_width =
            ElasticWordWidthV1::from_bits(target_bits).map_err(|error| error.to_string())?;
        if target_width.bits() < observed_floor || target_width.bits() > config.max_width_bits {
            return Err("planner produced an inadmissible width".to_owned());
        }
        add_duration(&mut stages.validation, validation_start);

        if target_bits != state.width_bits {
            let prior = plane.clone();
            let transition_start = Instant::now();
            let candidate = materialize(&plane, target_bits)?;
            add_duration(&mut stages.transition, transition_start);
            counters.transitions += 1;

            let verify_start = Instant::now();
            verify_payload(&plane, &candidate)?;
            let injected_failure = config.rollback_epoch == Some(epoch);
            add_duration(&mut stages.verification, verify_start);

            if injected_failure {
                let rollback_start = Instant::now();
                let restored = materialize(&candidate, state.width_bits)?;
                verify_payload(&prior, &restored)?;
                add_duration(&mut stages.rollback, rollback_start);
                counters.rollbacks += 1;

                let retry_start = Instant::now();
                plane = materialize(&restored, target_bits)?;
                add_duration(&mut stages.transition, retry_start);
                counters.retries += 1;

                let retry_verify_start = Instant::now();
                verify_payload(&restored, &plane)?;
                add_duration(&mut stages.verification, retry_verify_start);
            } else {
                plane = candidate;
            }

            state.width_bits = target_bits;
            state.epochs_since_transition = 0;
        } else {
            state.epochs_since_transition = state.epochs_since_transition.saturating_add(1);
        }

        let traversal_start = Instant::now();
        checksum ^= black_box(touch_plane(&plane));
        add_duration(&mut stages.traversal, traversal_start);
    }

    Ok((
        sequence_start.elapsed().as_nanos(),
        stages,
        counters,
        checksum,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StaticResult {
    full_sequence_admissible: bool,
    blocked_epochs: usize,
    elapsed_ns: u128,
    traversal_ns: u128,
    checksum: u64,
}

fn run_static_once(config: Config, width_bits: u16) -> Result<StaticResult, String> {
    let width = ElasticWordWidthV1::from_bits(width_bits).map_err(|error| error.to_string())?;
    let plane = base_plane(config.words, width_bits, config.seed)?;
    let start = Instant::now();
    let mut traversal_ns = 0_u128;
    let mut checksum = 0_u64;
    let mut blocked_epochs = 0_usize;

    for floor_bits in DEMAND_BITS {
        if width_bits < floor_bits || width_bits > config.max_width_bits {
            blocked_epochs += 1;
            continue;
        }
        let traversal_start = Instant::now();
        checksum ^= black_box(touch_plane(&plane));
        add_duration(&mut traversal_ns, traversal_start);
        black_box(width);
    }

    Ok(StaticResult {
        full_sequence_admissible: blocked_epochs == 0,
        blocked_epochs,
        elapsed_ns: start.elapsed().as_nanos(),
        traversal_ns,
        checksum,
    })
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2
    } else {
        values[middle]
    }
}

fn run(config: Config) -> Result<(), String> {
    println!(
        "schema,arm,width_bits,full_sequence_admissible,blocked_epochs,words,epochs,max_width_bits,cooldown_epochs,rollback_epoch,median_total_ns,median_observation_ns,median_planning_ns,median_validation_ns,median_transition_ns,median_verification_ns,median_rollback_ns,median_traversal_ns,transitions,rollbacks,retries,checksum"
    );

    for static_width in WIDTHS {
        let mut total = Vec::with_capacity(config.repetitions);
        let mut traversal = Vec::with_capacity(config.repetitions);
        let mut last = None;
        for _ in 0..config.repetitions {
            let result = run_static_once(config, static_width)?;
            total.push(result.elapsed_ns);
            traversal.push(result.traversal_ns);
            last = Some(result);
        }
        let result = last.expect("non-empty repetitions");
        println!(
            "{SCHEMA},static,{static_width},{},{},{},{},{},{},{}, {},0,0,0,0,0,0,{},0,0,0,{}",
            result.full_sequence_admissible,
            result.blocked_epochs,
            config.words,
            DEMAND_BITS.len(),
            config.max_width_bits,
            config.cooldown_epochs,
            config
                .rollback_epoch
                .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            median(&mut total),
            median(&mut traversal),
            result.checksum
        );
    }

    let mut totals = Vec::with_capacity(config.repetitions);
    let mut observation = Vec::with_capacity(config.repetitions);
    let mut planning = Vec::with_capacity(config.repetitions);
    let mut validation = Vec::with_capacity(config.repetitions);
    let mut transition = Vec::with_capacity(config.repetitions);
    let mut verification = Vec::with_capacity(config.repetitions);
    let mut rollback = Vec::with_capacity(config.repetitions);
    let mut traversal = Vec::with_capacity(config.repetitions);
    let mut last_counters = AdaptiveCounters::default();
    let mut last_checksum = 0_u64;

    for _ in 0..config.repetitions {
        let (total, stages, counters, checksum) = run_adaptive_once(config)?;
        totals.push(total);
        observation.push(stages.observation);
        planning.push(stages.planning);
        validation.push(stages.validation);
        transition.push(stages.transition);
        verification.push(stages.verification);
        rollback.push(stages.rollback);
        traversal.push(stages.traversal);
        last_counters = counters;
        last_checksum = checksum;
    }

    println!(
        "{SCHEMA},adaptive,elastic,true,0,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        config.words,
        DEMAND_BITS.len(),
        config.max_width_bits,
        config.cooldown_epochs,
        config
            .rollback_epoch
            .map_or_else(|| "none".to_owned(), |value| value.to_string()),
        median(&mut totals),
        median(&mut observation),
        median(&mut planning),
        median(&mut validation),
        median(&mut transition),
        median(&mut verification),
        median(&mut rollback),
        median(&mut traversal),
        last_counters.transitions,
        last_counters.rollbacks,
        last_counters.retries,
        last_checksum
    );

    Ok(())
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!(
                "usage: eww_adaptive_policy_bench <words> <repetitions> <cooldown_epochs> <max_width_bits> <rollback_epoch_or_max> <seed>"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            words: 8,
            repetitions: 1,
            cooldown_epochs: 2,
            max_width_bits: 512,
            rollback_epoch: None,
            seed: 42,
        }
    }

    #[test]
    fn policy_expands_immediately_and_contracts_after_cooldown() {
        let mut state = PolicyState {
            width_bits: 64,
            epochs_since_transition: 0,
        };
        assert_eq!(choose_width(state, 256, 2, 512).unwrap(), 256);

        state = PolicyState {
            width_bits: 256,
            epochs_since_transition: 1,
        };
        assert_eq!(choose_width(state, 64, 2, 512).unwrap(), 256);

        state.epochs_since_transition = 2;
        assert_eq!(choose_width(state, 64, 2, 512).unwrap(), 64);
    }

    #[test]
    fn static_w512_is_admissible_and_w256_is_not_for_frozen_trace() {
        let cfg = config();
        let w512 = run_static_once(cfg, 512).unwrap();
        let w256 = run_static_once(cfg, 256).unwrap();
        assert!(w512.full_sequence_admissible);
        assert_eq!(w512.blocked_epochs, 0);
        assert!(!w256.full_sequence_admissible);
        assert_eq!(
            w256.blocked_epochs,
            DEMAND_BITS.iter().filter(|bits| **bits > 256).count()
        );
    }

    #[test]
    fn adaptive_trace_runs_under_w512_budget() {
        let (elapsed, _stages, counters, _checksum) = run_adaptive_once(config()).unwrap();
        assert!(elapsed > 0);
        assert!(counters.transitions > 0);
        assert_eq!(counters.rollbacks, 0);
    }

    #[test]
    fn deterministic_fault_exercises_rollback_and_retry() {
        let mut cfg = config();
        cfg.rollback_epoch = Some(2);
        let (_elapsed, stages, counters, _checksum) = run_adaptive_once(cfg).unwrap();
        assert_eq!(counters.rollbacks, 1);
        assert_eq!(counters.retries, 1);
        assert!(stages.rollback > 0);
    }

    #[test]
    fn budget_below_observed_floor_fails_closed() {
        let mut cfg = config();
        cfg.max_width_bits = 256;
        let error = run_adaptive_once(cfg).unwrap_err();
        assert!(error.contains("exceeds memory-budget width"));
    }
}
