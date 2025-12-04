//! Binary for listing available datasets in the registry.

use clap::Parser;
use rank_fusion_evals::dataset_registry::{DatasetCategory, DatasetRegistry};

#[derive(Parser)]
#[command(name = "list-datasets")]
#[command(about = "List available evaluation datasets")]
struct Args {
    /// Filter by priority (1-7)
    #[arg(long)]
    priority: Option<u8>,

    /// Filter by category
    #[arg(long)]
    category: Option<String>,

    /// Show detailed information
    #[arg(long, short = 'd')]
    detailed: bool,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let registry = DatasetRegistry::new();

    let datasets = if let Some(priority) = args.priority {
        registry.by_priority(priority)
    } else if let Some(category_str) = &args.category {
        let category = match category_str.as_str() {
            "general" => DatasetCategory::General,
            "multilingual" => DatasetCategory::Multilingual,
            "domain" | "domain-specific" => DatasetCategory::DomainSpecific,
            "qa" | "question-answering" => DatasetCategory::QuestionAnswering,
            "regional" => DatasetCategory::Regional,
            "specialized" => DatasetCategory::Specialized,
            _ => {
                eprintln!("Unknown category: {}. Use: general, multilingual, domain, qa, regional, specialized", category_str);
                std::process::exit(1);
            }
        };
        registry.by_category(category)
    } else {
        registry.all()
    };

    if args.json {
        let json = serde_json::to_string_pretty(&datasets)?;
        println!("{}", json);
    } else if args.detailed {
        for dataset in datasets {
            println!("\n╔════════════════════════════════════════════════════════════════╗");
            println!("║ {:<60} ║", dataset.name);
            println!("╠════════════════════════════════════════════════════════════════╣");
            println!("║ Priority: {}                                                  ║", dataset.priority);
            println!("║ Category: {:<50} ║", format!("{:?}", dataset.category));
            println!("║ Description: {:<47} ║", dataset.description);
            if let Some(domain) = &dataset.domain {
                println!("║ Domain: {:<54} ║", domain);
            }
            if let Some(queries) = dataset.queries {
                println!("║ Queries: {:<52} ║", format!("{}", queries));
            }
            if let Some(docs) = dataset.documents {
                println!("║ Documents: {:<50} ║", format!("{}", docs));
            }
            println!("║ Languages: {:<50} ║", dataset.languages.join(", "));
            if let Some(url) = &dataset.url {
                println!("║ URL: {:<56} ║", url);
            }
            if let Some(notes) = &dataset.notes {
                println!("║ Notes: {:<54} ║", notes);
            }
            println!("╚════════════════════════════════════════════════════════════════╝");
        }
    } else {
        println!("Available Datasets ({} total)\n", datasets.len());
        println!("{:<30} {:>8} {:>20} {:>15}", "Name", "Priority", "Category", "Queries");
        println!("{}", "-".repeat(80));
        for dataset in datasets {
            let queries_str = dataset
                .queries
                .map(|q| format!("{}", q))
                .unwrap_or_else(|| "N/A".to_string());
            println!(
                "{:<30} {:>8} {:>20} {:>15}",
                dataset.name,
                dataset.priority,
                format!("{:?}", dataset.category),
                queries_str
            );
        }
    }

    Ok(())
}

