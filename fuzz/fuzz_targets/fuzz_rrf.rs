#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rank_fusion::{rrf_with_config, RrfConfig};

#[derive(Arbitrary, Debug)]
struct Input {
    list_a: Vec<(u32, f32)>,
    list_b: Vec<(u32, f32)>,
    k: u32,
}

fuzz_target!(|input: Input| {
    // Bound k to reasonable range
    let k = input.k.saturating_add(1).min(10000);
    let config = RrfConfig::new(k);

    // Should not panic
    let _ = rrf_with_config(&input.list_a, &input.list_b, config);
});

