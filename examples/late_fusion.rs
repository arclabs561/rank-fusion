//! Late fusion for multi-modal search.
//!
//! Combine results from different modalities:
//! text, image, audio, structured data.
//!
//! Run: `cargo run --example late_fusion`

use rank_fusion::{dbsf, rrf, weighted, WeightedConfig};

fn main() {
    println!("=== Multi-Modal Late Fusion ===\n");

    let query = "red sports car on mountain road";
    println!("Query: \"{query}\"\n");

    // Text search results (caption/description matching)
    let text_results: Vec<(u32, f32)> = vec![
        (1, 0.92), // "Ferrari on mountain pass"
        (2, 0.88), // "Porsche driving mountain roads"
        (3, 0.85), // "Red car advertisement"
        (4, 0.75), // "Mountain landscape with road"
        (5, 0.70), // "Sports car review"
    ];

    // Image search results (visual similarity)
    let image_results: Vec<(u32, f32)> = vec![
        (4, 0.95), // Mountain with road (visually matches)
        (1, 0.90), // Ferrari image
        (6, 0.85), // Red car parked
        (2, 0.80), // Porsche image
        (7, 0.75), // Mountain scenery
    ];

    // Metadata search (tags, attributes)
    let metadata_results: Vec<(u32, f32)> = vec![
        (2, 0.90), // Tagged: sports-car, mountain, red
        (1, 0.85), // Tagged: ferrari, mountain
        (6, 0.80), // Tagged: red, car
        (3, 0.70), // Tagged: red, advertisement
        (8, 0.65), // Tagged: sports, vehicle
    ];

    println!(
        "Text results:    {:?}",
        text_results.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!(
        "Image results:   {:?}",
        image_results.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!(
        "Metadata results:{:?}",
        metadata_results.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );

    // Problem: Different score scales
    println!("\nScore ranges:");
    println!(
        "  Text:     [{:.2}, {:.2}]",
        text_results.last().unwrap().1,
        text_results[0].1
    );
    println!(
        "  Image:    [{:.2}, {:.2}]",
        image_results.last().unwrap().1,
        image_results[0].1
    );
    println!(
        "  Metadata: [{:.2}, {:.2}]",
        metadata_results.last().unwrap().1,
        metadata_results[0].1
    );

    // Solution 1: RRF (rank-based, ignores scores)
    let rrf_text_img = rrf(&text_results, &image_results);
    let rrf_all = rrf(&rrf_text_img, &metadata_results);

    println!("\nRRF (rank-based):");
    for (id, score) in rrf_all.iter().take(5) {
        println!("  item_{id}: {score:.4}");
    }

    // Solution 2: DBSF (distribution-based, normalizes scores)
    let dbsf_text_img = dbsf(&text_results, &image_results);
    let dbsf_all = dbsf(&dbsf_text_img, &metadata_results);

    println!("\nDBSF (score-normalized):");
    for (id, score) in dbsf_all.iter().take(5) {
        println!("  item_{id}: {score:.4}");
    }

    // Solution 3: Weighted (when modality reliability known)
    // Image is most reliable for this visual query
    let weighted_ti = weighted(
        &text_results,
        &image_results,
        WeightedConfig::new(0.3, 0.7),
    );
    let weighted_all = weighted(
        &weighted_ti,
        &metadata_results,
        WeightedConfig::new(0.8, 0.2),
    );

    println!("\nWeighted (image-first):");
    for (id, score) in weighted_all.iter().take(5) {
        println!("  item_{id}: {score:.4}");
    }

    // Summary:
    // - RRF uses only rank positions, ignoring score values entirely
    // - DBSF normalizes by z-score (subtract mean, divide by stddev)
    // - Weighted assumes you've calibrated relative modality quality
}
