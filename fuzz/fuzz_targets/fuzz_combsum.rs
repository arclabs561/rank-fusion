#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rank_fusion::combsum;

#[derive(Arbitrary, Debug)]
struct Input {
    list_a: Vec<(u32, f32)>,
    list_b: Vec<(u32, f32)>,
}

fuzz_target!(|input: Input| {
    // Should not panic on any input
    let _ = combsum(&input.list_a, &input.list_b);
});

