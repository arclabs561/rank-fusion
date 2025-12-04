//! Evaluation runner for rank-fusion methods.
//!
//! Runs all fusion methods on all scenarios and generates an HTML report.

mod datasets;
mod metrics;
mod real_world;
mod dataset_loaders;

use datasets::{all_scenarios, Scenario};
use metrics::Metrics;
use rank_fusion::{
    additive_multi_task_with_config, borda, combmnz, combsum, dbsf, isr, rrf,
    standardized_with_config, weighted, AdditiveMultiTaskConfig, StandardizedConfig,
    WeightedConfig,
};
use std::collections::HashMap;
use std::collections::HashSet;

/// Run all fusion methods on a scenario.
fn evaluate_scenario(scenario: &Scenario) -> HashMap<&'static str, (Vec<String>, Metrics)> {
    let a = &scenario.retriever_a;
    let b = &scenario.retriever_b;

    let methods: Vec<(&str, Vec<(String, f32)>)> = vec![
        ("rrf", rrf(a, b)),
        ("isr", isr(a, b)),
        ("combsum", combsum(a, b)),
        ("combmnz", combmnz(a, b)),
        ("borda", borda(a, b)),
        ("dbsf", dbsf(a, b)),
        (
            "weighted_0.7",
            weighted(a, b, WeightedConfig::new(0.7, 0.3)),
        ),
        (
            "weighted_0.9",
            weighted(a, b, WeightedConfig::new(0.9, 0.1)),
        ),
        (
            "standardized",
            standardized_with_config(a, b, StandardizedConfig::default()),
        ),
        (
            "standardized_tight",
            standardized_with_config(a, b, StandardizedConfig::new((-2.0, 2.0))),
        ),
        (
            "additive_multi_task_1_1",
            additive_multi_task_with_config(a, b, AdditiveMultiTaskConfig::new((1.0, 1.0))),
        ),
        (
            "additive_multi_task_1_20",
            additive_multi_task_with_config(a, b, AdditiveMultiTaskConfig::new((1.0, 20.0))),
        ),
    ];

    methods
        .into_iter()
        .map(|(name, fused)| {
            let ranked: Vec<String> = fused.into_iter().map(|(id, _)| id).collect();
            let metrics = Metrics::compute(&ranked, &scenario.relevant);
            (name, (ranked, metrics))
        })
        .collect()
}

/// Result for a single scenario.
#[derive(serde::Serialize)]
struct ScenarioResult {
    name: String,
    description: String,
    insight: String,
    expected_winner: String,
    actual_winner: String,
    correct: bool,
    methods: HashMap<String, MethodResult>,
}

#[derive(serde::Serialize)]
struct MethodResult {
    ranking: Vec<String>,
    metrics: Metrics,
}

/// Generate HTML report.
fn generate_html(results: &[ScenarioResult]) -> String {
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Rank Fusion Evaluation Report</title>
    <style>
        body { 
            font-family: 'SF Mono', 'Menlo', monospace; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; border-bottom: 2px solid #00d9ff; padding-bottom: 10px; }
        h2 { color: #ff6b9d; margin-top: 40px; }
        .scenario { 
            background: #16213e; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            border-left: 4px solid #00d9ff;
        }
        .correct { border-left-color: #00ff88; }
        .incorrect { border-left-color: #ff4757; }
        .description { color: #888; font-style: italic; margin-bottom: 10px; }
        .insight { color: #ffd93d; background: #0f3460; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0;
            font-size: 13px;
        }
        th, td { 
            border: 1px solid #333; 
            padding: 8px 12px; 
            text-align: left; 
        }
        th { 
            background: #0f3460; 
            color: #00d9ff;
        }
        tr:nth-child(even) { background: #1a1a2e; }
        tr:hover { background: #0f3460; }
        .winner { background: #00ff88 !important; color: #000; font-weight: bold; }
        .metric-best { color: #00ff88; font-weight: bold; }
        .ranking { font-family: monospace; color: #ffd93d; }
        .summary { 
            background: #0f3460; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 30px;
        }
        .stat { display: inline-block; margin-right: 30px; }
        .stat-value { font-size: 2em; color: #00ff88; }
        .stat-label { color: #888; }
    </style>
</head>
<body>
    <h1>Rank Fusion Evaluation Report</h1>
"#,
    );

    // Summary statistics
    let correct = results.iter().filter(|r| r.correct).count();
    let total = results.len();
    html.push_str(&format!(
        r#"
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{}/{}</div>
            <div class="stat-label">Scenarios Correct</div>
        </div>
        <div class="stat">
            <div class="stat-value">{:.0}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
"#,
        correct,
        total,
        100.0 * correct as f64 / total as f64
    ));

    // Each scenario
    for result in results {
        let class = if result.correct {
            "correct"
        } else {
            "incorrect"
        };
        html.push_str(&format!(
            r#"
    <div class="scenario {}">
        <h2>{}</h2>
        <p class="description">{}</p>
        <p class="insight"><strong>Insight:</strong> {}</p>
        <p>Expected winner: <strong>{}</strong> | Actual winner: <strong>{}</strong></p>
        
        <table>
            <tr>
                <th>Method</th>
                <th>Top-5 Ranking</th>
                <th>P@1</th>
                <th>P@5</th>
                <th>MRR</th>
                <th>nDCG@5</th>
                <th>nDCG@10</th>
                <th>AP</th>
            </tr>
"#,
            class,
            result.name,
            result.description,
            result.insight,
            result.expected_winner,
            result.actual_winner
        ));

        // Find best values for highlighting
        let best_ndcg5 = result
            .methods
            .values()
            .map(|m| m.metrics.ndcg_at_5)
            .fold(0.0, f64::max);

        for (method, data) in &result.methods {
            let is_winner = *method == result.actual_winner;
            let row_class = if is_winner { "winner" } else { "" };
            let ranking_str = data
                .ranking
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(" → ");
            let ndcg_class = if (data.metrics.ndcg_at_5 - best_ndcg5).abs() < 1e-9 {
                "metric-best"
            } else {
                ""
            };

            html.push_str(&format!(
                r#"            <tr class="{}">
                <td>{}</td>
                <td class="ranking">{}</td>
                <td>{:.3}</td>
                <td>{:.3}</td>
                <td>{:.3}</td>
                <td class="{}">{:.3}</td>
                <td>{:.3}</td>
                <td>{:.3}</td>
            </tr>
"#,
                row_class,
                method,
                ranking_str,
                data.metrics.precision_at_1,
                data.metrics.precision_at_5,
                data.metrics.mrr,
                ndcg_class,
                data.metrics.ndcg_at_5,
                data.metrics.ndcg_at_10,
                data.metrics.average_precision
            ));
        }

        html.push_str("        </table>\n    </div>\n");
    }

    html.push_str(
        r#"
    <h2>Method Descriptions</h2>
    <table>
        <tr><th>Method</th><th>Formula</th><th>Best For</th></tr>
        <tr><td>rrf</td><td>Σ 1/(k + rank)</td><td>Different score scales</td></tr>
        <tr><td>isr</td><td>Σ 1/sqrt(k + rank)</td><td>When lower ranks matter</td></tr>
        <tr><td>combsum</td><td>Σ normalized_score</td><td>Same scale, trust scores</td></tr>
        <tr><td>combmnz</td><td>Σ score × overlap_count</td><td>Reward overlap</td></tr>
        <tr><td>borda</td><td>Σ (N - rank)</td><td>Voting/committee</td></tr>
        <tr><td>dbsf</td><td>Σ z_score</td><td>Different distributions</td></tr>
        <tr><td>weighted</td><td>w₁×s₁ + w₂×s₂</td><td>Known retriever quality</td></tr>
        <tr><td>standardized</td><td>Σ z_score (configurable clip)</td><td>ERANK-style, robust to outliers</td></tr>
        <tr><td>additive_multi_task</td><td>α·A + β·B</td><td>ResFlow-style, e-commerce ranking</td></tr>
    </table>
</body>
</html>
"#,
    );

    html
}

/// Print comprehensive CLI summary with deep analysis.
fn print_comprehensive_summary(results: &[ScenarioResult]) {
    let total = results.len();
    let correct = results.iter().filter(|r| r.correct).count();
    let accuracy = 100.0 * correct as f64 / total as f64;

    // ─────────────────────────────────────────────────────────────────────────
    // Header
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    Rank Fusion Evaluation Results                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Overall Summary
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Overall Summary ───────────────────────────────────────────────────────────┐");
    println!("│ Scenarios: {:<3} total  │  Correct: {:<3} ({:>5.1}%)  │  Failed: {:<3} ({:>5.1}%) │",
        total, correct, accuracy, total - correct, 100.0 - accuracy);
    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Method Win Statistics & Performance Collection
    // ─────────────────────────────────────────────────────────────────────────
    let mut method_wins: HashMap<String, usize> = HashMap::new();
    let mut method_ndcg5: HashMap<String, Vec<f64>> = HashMap::new();
    let mut method_ap: HashMap<String, Vec<f64>> = HashMap::new();
    let mut method_p1: HashMap<String, Vec<f64>> = HashMap::new();
    let mut method_mrr: HashMap<String, Vec<f64>> = HashMap::new();
    let mut method_ndcg10: HashMap<String, Vec<f64>> = HashMap::new();

    for result in results {
        // Count wins
        *method_wins.entry(result.actual_winner.clone()).or_insert(0) += 1;

        // Collect metrics for each method across all scenarios
        for (method_name, method_result) in &result.methods {
            method_ndcg5.entry(method_name.clone()).or_insert_with(Vec::new)
                .push(method_result.metrics.ndcg_at_5);
            method_ap.entry(method_name.clone()).or_insert_with(Vec::new)
                .push(method_result.metrics.average_precision);
            method_p1.entry(method_name.clone()).or_insert_with(Vec::new)
                .push(method_result.metrics.precision_at_1);
            method_mrr.entry(method_name.clone()).or_insert_with(Vec::new)
                .push(method_result.metrics.mrr);
            method_ndcg10.entry(method_name.clone()).or_insert_with(Vec::new)
                .push(method_result.metrics.ndcg_at_10);
        }
    }

    // Calculate statistics for each method (including all methods, not just winners)
    let mut method_stats: Vec<(String, usize, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
    let empty_vec = Vec::<f64>::new();
    
    // Get all methods that appear in any scenario
    let mut all_methods = HashSet::new();
    for result in results {
        for method_name in result.methods.keys() {
            all_methods.insert(method_name.clone());
        }
    }
    
    for method in &all_methods {
        let wins = method_wins.get(method).copied().unwrap_or(0);
        let ndcg5_values = method_ndcg5.get(method).unwrap_or(&empty_vec);
        let ap_values = method_ap.get(method).unwrap_or(&empty_vec);
        let p1_values = method_p1.get(method).unwrap_or(&empty_vec);
        let mrr_values = method_mrr.get(method).unwrap_or(&empty_vec);
        let ndcg10_values = method_ndcg10.get(method).unwrap_or(&empty_vec);

        let avg_ndcg5 = if ndcg5_values.is_empty() { 0.0 } else {
            ndcg5_values.iter().sum::<f64>() / ndcg5_values.len() as f64
        };
        let avg_ap = if ap_values.is_empty() { 0.0 } else {
            ap_values.iter().sum::<f64>() / ap_values.len() as f64
        };
        let avg_p1 = if p1_values.is_empty() { 0.0 } else {
            p1_values.iter().sum::<f64>() / p1_values.len() as f64
        };
        let avg_mrr = if mrr_values.is_empty() { 0.0 } else {
            mrr_values.iter().sum::<f64>() / mrr_values.len() as f64
        };
        let avg_ndcg10 = if ndcg10_values.is_empty() { 0.0 } else {
            ndcg10_values.iter().sum::<f64>() / ndcg10_values.len() as f64
        };
        let min_ndcg5 = if ndcg5_values.is_empty() { 0.0 } else {
            ndcg5_values.iter().fold(1.0_f64, |a, &b| a.min(b))
        };
        let max_ndcg5 = if ndcg5_values.is_empty() { 0.0 } else {
            ndcg5_values.iter().fold(0.0_f64, |a, &b| a.max(b))
        };

        method_stats.push((
            method.clone(),
            wins,
            avg_ndcg5,
            avg_ap,
            avg_p1,
            avg_mrr,
            avg_ndcg10,
            min_ndcg5,
            max_ndcg5,
        ));
    }

    // Sort by wins, then by average nDCG@5
    method_stats.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    println!("┌─ Method Performance Summary (All Methods) ────────────────────────────────────────┐");
    println!("│ {:<28} │ {:>4} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │",
        "Method", "Wins", "nDCG@5", "nDCG@10", "AP", "P@1", "MRR", "Min", "Max");
    println!("├──────────────────────────────┼──────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤");
    for (method, wins, avg_ndcg5, avg_ap, avg_p1, avg_mrr, avg_ndcg10, min_ndcg5, max_ndcg5) in &method_stats {
        let win_indicator = if *wins > 0 { "★" } else { " " };
        println!("│ {}{:<27} │ {:>4} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>7.3} │",
            win_indicator, method, wins, avg_ndcg5, avg_ndcg10, avg_ap, avg_p1, avg_mrr, min_ndcg5, max_ndcg5);
    }
    println!("└──────────────────────────────┴──────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘");
    println!("│ ★ = Method won at least one scenario");
    println!("└────────────────────────────────────────────────────────────────────────────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Scenario Results (Grouped)
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Scenario Results ─────────────────────────────────────────────────────────────┐");
    
    // Group scenarios by correctness
    let mut correct_scenarios = Vec::new();
    let mut incorrect_scenarios = Vec::new();
    
    for result in results {
        if result.correct {
            correct_scenarios.push(result);
        } else {
            incorrect_scenarios.push(result);
        }
    }

    // Print correct scenarios (compact)
    if !correct_scenarios.is_empty() {
        println!("│ ✓ Correct Scenarios ({})", correct_scenarios.len());
        for result in &correct_scenarios {
            let winner_ndcg = result.methods.get(&result.actual_winner)
                .map(|m| m.metrics.ndcg_at_5)
                .unwrap_or(0.0);
            println!("│   ✓ {:<35} → {} (nDCG@5: {:.3})",
                result.name, result.actual_winner, winner_ndcg);
        }
    }

    // Print incorrect scenarios (detailed)
    if !incorrect_scenarios.is_empty() {
        println!("│");
        println!("│ ✗ Failed Scenarios ({})", incorrect_scenarios.len());
        for result in &incorrect_scenarios {
            let expected_ndcg = result.methods.get(&result.expected_winner)
                .map(|m| m.metrics.ndcg_at_5)
                .unwrap_or(0.0);
            let actual_ndcg = result.methods.get(&result.actual_winner)
                .map(|m| m.metrics.ndcg_at_5)
                .unwrap_or(0.0);
            println!("│   ✗ {:<35}", result.name);
            println!("│     Expected: {} (nDCG@5: {:.3})", result.expected_winner, expected_ndcg);
            println!("│     Actual:   {} (nDCG@5: {:.3})", result.actual_winner, actual_ndcg);
            println!("│     Insight:  {}", result.insight);
        }
    }
    
    println!("└────────────────────────────────────────────────────────────────────────────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Top Methods Analysis (by different metrics)
    // ─────────────────────────────────────────────────────────────────────────
    if !method_stats.is_empty() {
        println!("┌─ Top Performing Methods ─────────────────────────────────────────────────────┐");
        
        // Top by wins
        let mut by_wins = method_stats.clone();
        by_wins.sort_by(|a, b| b.1.cmp(&a.1));
        println!("│ Top 3 by Wins:");
        for (i, (method, wins, avg_ndcg5, avg_ap, _, _, _, _, _)) in by_wins.iter().take(3).enumerate() {
            let rank_symbol = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "  ",
            };
            println!("│   {} {:<25} │ Wins: {:>3} │ Avg nDCG@5: {:.3} │ Avg AP: {:.3} │",
                rank_symbol, method, wins, avg_ndcg5, avg_ap);
        }
        
        // Top by average nDCG@5
        let mut by_ndcg5 = method_stats.clone();
        by_ndcg5.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        println!("│ Top 3 by Average nDCG@5:");
        for (i, (method, wins, avg_ndcg5, avg_ap, _, _, _, _, _)) in by_ndcg5.iter().take(3).enumerate() {
            let rank_symbol = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "  ",
            };
            println!("│   {} {:<25} │ nDCG@5: {:.3} │ Wins: {:>3} │ Avg AP: {:.3} │",
                rank_symbol, method, avg_ndcg5, wins, avg_ap);
        }
        
        println!("└────────────────────────────────────────────────────────────────────────────────┘\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Metric Distribution Analysis
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Metric Distribution Analysis ──────────────────────────────────────────────────┐");
    
    // Calculate overall statistics
    let all_ndcg5: Vec<f64> = results.iter()
        .flat_map(|r| r.methods.values().map(|m| m.metrics.ndcg_at_5))
        .collect();
    
    if !all_ndcg5.is_empty() {
        let mean_ndcg5 = all_ndcg5.iter().sum::<f64>() / all_ndcg5.len() as f64;
        let variance = all_ndcg5.iter()
            .map(|x| (x - mean_ndcg5).powi(2))
            .sum::<f64>() / all_ndcg5.len() as f64;
        let std_dev = variance.sqrt();
        let min_ndcg5 = all_ndcg5.iter().fold(1.0_f64, |a, &b| a.min(b));
        let max_ndcg5 = all_ndcg5.iter().fold(0.0_f64, |a, &b| a.max(b));
        
        // Percentiles
        let mut sorted = all_ndcg5.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p25 = sorted[sorted.len() / 4];
        let p50 = sorted[sorted.len() / 2];
        let p75 = sorted[3 * sorted.len() / 4];

        println!("│ nDCG@5 Statistics (across all methods and scenarios):");
        println!("│   Mean: {:.4}  │  Std Dev: {:.4}  │  Min: {:.4}  │  Max: {:.4}",
            mean_ndcg5, std_dev, min_ndcg5, max_ndcg5);
        println!("│   P25: {:.4}  │  Median: {:.4}  │  P75: {:.4}",
            p25, p50, p75);
    }

    println!("└────────────────────────────────────────────────────────────────────────────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Method Consistency Analysis
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Method Consistency (Standard Deviation) ──────────────────────────────────────┐");
    println!("│ Lower std dev = more consistent performance across scenarios                     │");
    println!("│ {:<28} │ {:>7} │ {:>7} │ {:>7} │",
        "Method", "nDCG@5 σ", "AP σ", "P@1 σ");
    println!("├──────────────────────────────┼─────────┼─────────┼─────────┤");
    
    let mut consistency_stats: Vec<(String, f64, f64, f64)> = Vec::new();
    let empty_vec = Vec::<f64>::new();
    for (method, _) in &method_wins {
        let ndcg5_vals = method_ndcg5.get(method).unwrap_or(&empty_vec);
        let ap_vals = method_ap.get(method).unwrap_or(&empty_vec);
        let p1_vals = method_p1.get(method).unwrap_or(&empty_vec);

        let calc_std = |vals: &[f64]| -> f64 {
            if vals.len() < 2 { return 0.0; }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            variance.sqrt()
        };

        let ndcg5_std = calc_std(ndcg5_vals);
        let ap_std = calc_std(ap_vals);
        let p1_std = calc_std(p1_vals);

        consistency_stats.push((method.clone(), ndcg5_std, ap_std, p1_std));
    }

    consistency_stats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for (method, ndcg5_std, ap_std, p1_std) in &consistency_stats {
        println!("│ {:<28} │ {:>7.4} │ {:>7.4} │ {:>7.4} │",
            method, ndcg5_std, ap_std, p1_std);
    }
    println!("└──────────────────────────────┴─────────┴─────────┴─────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Scenario-by-Scenario Breakdown (Compact)
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Detailed Scenario Breakdown ────────────────────────────────────────────────────┐");
    for (i, result) in results.iter().enumerate() {
        let status = if result.correct { "✓" } else { "✗" };
        let winner_ndcg = result.methods.get(&result.actual_winner)
            .map(|m| m.metrics.ndcg_at_5)
            .unwrap_or(0.0);
        let winner_ap = result.methods.get(&result.actual_winner)
            .map(|m| m.metrics.average_precision)
            .unwrap_or(0.0);
        let winner_mrr = result.methods.get(&result.actual_winner)
            .map(|m| m.metrics.mrr)
            .unwrap_or(0.0);
        
        // Find best nDCG@5 in this scenario for comparison
        let best_ndcg = result.methods.values()
            .map(|m| m.metrics.ndcg_at_5)
            .fold(0.0_f64, f64::max);
        let gap = best_ndcg - winner_ndcg;
        let gap_str = if gap < 1e-9 { "=" } else { "!" };
        
        println!("│ [{:>2}] {} {:<30} │ Winner: {:<25} │ nDCG@5: {:.3} {} │ AP: {:.3} │ MRR: {:.3} │",
            i + 1, status, result.name, result.actual_winner, winner_ndcg, gap_str, winner_ap, winner_mrr);
    }
    println!("└────────────────────────────────────────────────────────────────────────────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Method Comparison Matrix
    // ─────────────────────────────────────────────────────────────────────────
    println!("┌─ Method Comparison: Wins vs Average Performance ─────────────────────────────────┐");
    println!("│ Methods ranked by win count, showing average metrics across all scenarios        │");
    
    // Sort by wins, then show performance
    let mut by_wins = method_stats.clone();
    by_wins.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)));
    
    println!("│ {:<28} │ {:>4} │ {:>7} │ {:>7} │ {:>7} │",
        "Method", "Wins", "Avg nDCG@5", "Avg AP", "Avg MRR");
    println!("├──────────────────────────────┼──────┼─────────┼─────────┼─────────┤");
    for (method, wins, avg_ndcg5, avg_ap, _, avg_mrr, _, _, _) in &by_wins {
        if *wins > 0 || *avg_ndcg5 > 0.5 {
            println!("│ {:<28} │ {:>4} │ {:>7.3} │ {:>7.3} │ {:>7.3} │",
                method, wins, avg_ndcg5, avg_ap, avg_mrr);
        }
    }
    println!("└──────────────────────────────┴──────┴─────────┴─────────┴─────────┘\n");

    // ─────────────────────────────────────────────────────────────────────────
    // Footer
    // ─────────────────────────────────────────────────────────────────────────
    println!("Generated reports:");
    println!("  • eval_report.html  - Interactive HTML report");
    println!("  • eval_results.json  - Machine-readable JSON results");
    println!();
}

fn main() {
    let scenarios = all_scenarios();
    let mut results = Vec::new();

    for scenario in &scenarios {
        let evaluations = evaluate_scenario(scenario);

        // Find winner by nDCG@5, breaking ties with AP, then P@1, then MRR
        let best_ndcg = evaluations
            .values()
            .map(|(_, m)| m.ndcg_at_5)
            .fold(0.0, f64::max);

        // All methods achieving best nDCG
        let ndcg_winners: Vec<(&str, &Metrics)> = evaluations
            .iter()
            .filter(|(_, (_, m))| (m.ndcg_at_5 - best_ndcg).abs() < 1e-9)
            .map(|(name, (_, m))| (*name, m))
            .collect();

        // If multiple methods tie on nDCG@5, break tie with AP
        let best_ap = ndcg_winners
            .iter()
            .map(|(_, m)| m.average_precision)
            .fold(0.0, f64::max);

        let ap_winners: Vec<&str> = ndcg_winners
            .iter()
            .filter(|(_, m)| (m.average_precision - best_ap).abs() < 1e-9)
            .map(|(name, _)| *name)
            .collect();

        // If still tied, use P@1
        let best_p1 = evaluations
            .iter()
            .filter(|(name, _)| ap_winners.contains(name))
            .map(|(_, (_, m))| m.precision_at_1)
            .fold(0.0, f64::max);

        let winners: Vec<&str> = evaluations
            .iter()
            .filter(|(name, (_, m))| {
                ap_winners.contains(name)
                    && (m.precision_at_1 - best_p1).abs() < 1e-9
            })
            .map(|(name, _)| *name)
            .collect();

        // Primary winner (first alphabetically if tie)
        let winner = winners.iter().min().copied().unwrap_or("none");

        // Check if expected is among the tied winners
        let expected_wins = scenario.expected_winner == "all_equal"
            || winners
                .iter()
                .any(|w| *w == scenario.expected_winner || w.starts_with(scenario.expected_winner));

        let correct = expected_wins;

        let methods: HashMap<String, MethodResult> = evaluations
            .into_iter()
            .map(|(name, (ranking, metrics))| (name.to_string(), MethodResult { ranking, metrics }))
            .collect();

        results.push(ScenarioResult {
            name: scenario.name.to_string(),
            description: scenario.description.to_string(),
            insight: scenario.insight.to_string(),
            expected_winner: scenario.expected_winner.to_string(),
            actual_winner: winner.to_string(),
            correct,
            methods,
        });
    }

    // Print comprehensive CLI output
    print_comprehensive_summary(&results);

    // Generate HTML report
    let html = generate_html(&results);
    std::fs::write("eval_report.html", &html).expect("Failed to write HTML report");
    println!("\nHTML report written to eval_report.html");

    // Also write JSON for further analysis
    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write("eval_results.json", &json).expect("Failed to write JSON results");
    println!("JSON results written to eval_results.json");
}
