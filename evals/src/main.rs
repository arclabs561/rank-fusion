//! Evaluation runner for rank-fusion methods.
//!
//! Runs all fusion methods on all scenarios and generates an HTML report.

mod datasets;
mod metrics;

use datasets::{all_scenarios, Scenario};
use metrics::Metrics;
use rank_fusion::{borda, combmnz, combsum, dbsf, isr, rrf, weighted, WeightedConfig};
use std::collections::HashMap;

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
        let class = if result.correct { "correct" } else { "incorrect" };
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
            class, result.name, result.description, result.insight, result.expected_winner, result.actual_winner
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
            let ranking_str = data.ranking.iter().take(5).cloned().collect::<Vec<_>>().join(" → ");
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
    </table>
</body>
</html>
"#,
    );

    html
}

fn main() {
    let scenarios = all_scenarios();
    let mut results = Vec::new();

    for scenario in &scenarios {
        let evaluations = evaluate_scenario(scenario);

        // Find winner by nDCG@5, breaking ties with AP
        let best_ndcg = evaluations
            .values()
            .map(|(_, m)| m.ndcg_at_5)
            .fold(0.0, f64::max);
        
        // All methods achieving best nDCG
        let winners: Vec<&str> = evaluations
            .iter()
            .filter(|(_, (_, m))| (m.ndcg_at_5 - best_ndcg).abs() < 1e-9)
            .map(|(name, _)| *name)
            .collect();
        
        // Primary winner (first alphabetically if tie)
        let winner = winners.iter().min().copied().unwrap_or("none");
        
        // Check if expected is among the tied winners
        let expected_wins = scenario.expected_winner == "all_equal"
            || winners.iter().any(|w| *w == scenario.expected_winner || w.starts_with(scenario.expected_winner));
        
        let correct = expected_wins;

        let methods: HashMap<String, MethodResult> = evaluations
            .into_iter()
            .map(|(name, (ranking, metrics))| {
                (name.to_string(), MethodResult { ranking, metrics })
            })
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

    // Print summary
    println!("Rank Fusion Evaluation Results");
    println!("==============================\n");

    for result in &results {
        let status = if result.correct { "✓" } else { "✗" };
        println!(
            "{} {} - expected: {}, actual: {}",
            status, result.name, result.expected_winner, result.actual_winner
        );
    }

    let correct = results.iter().filter(|r| r.correct).count();
    println!("\nTotal: {}/{} scenarios correct", correct, results.len());

    // Generate HTML report
    let html = generate_html(&results);
    std::fs::write("eval_report.html", &html).expect("Failed to write HTML report");
    println!("\nHTML report written to eval_report.html");

    // Also write JSON for further analysis
    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write("eval_results.json", &json).expect("Failed to write JSON results");
    println!("JSON results written to eval_results.json");
}

