//! Dataset registry and metadata management.
//!
//! Provides a centralized registry of available datasets with metadata,
//! access information, and conversion requirements.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dataset registry entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    pub name: String,
    pub description: String,
    pub priority: u8, // 1-7
    pub category: DatasetCategory,
    pub languages: Vec<String>,
    pub domain: Option<String>,
    pub queries: Option<usize>,
    pub documents: Option<usize>,
    pub access_method: AccessMethod,
    pub format: DatasetFormat,
    pub url: Option<String>,
    pub citation: Option<String>,
    pub notes: Option<String>,
}

/// Dataset category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetCategory {
    General,
    Multilingual,
    DomainSpecific,
    QuestionAnswering,
    Regional,
    Specialized,
}

/// Access method for dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessMethod {
    HuggingFace { dataset_id: String },
    PythonFramework { package: String, function: String },
    DirectDownload { url: String },
    RegionalForum { website: String },
    RequiresAccess { contact: String },
}

/// Dataset format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetFormat {
    Trec, // Already in TREC format
    HuggingFace, // Needs conversion
    Beir, // BEIR format
    Custom { converter: Option<String> },
}

/// Dataset registry.
pub struct DatasetRegistry {
    datasets: HashMap<String, DatasetEntry>,
}

impl DatasetRegistry {
    /// Create a new registry with all known datasets.
    pub fn new() -> Self {
        let mut registry = DatasetRegistry {
            datasets: HashMap::new(),
        };

        // Priority 1: Essential
        registry.add_dataset(DatasetEntry {
            name: "msmarco-passage".to_string(),
            description: "MS MARCO Passage Ranking - Industry standard, large-scale".to_string(),
            priority: 1,
            category: DatasetCategory::General,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(124_000),
            documents: Some(8_800_000),
            access_method: AccessMethod::DirectDownload {
                url: "https://microsoft.github.io/msmarco/".to_string(),
            },
            format: DatasetFormat::Trec,
            url: Some("https://microsoft.github.io/msmarco/".to_string()),
            citation: None,
            notes: Some("Use MS MARCO v2 (cleaner, less biased)".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "beir".to_string(),
            description: "BEIR - 13 public datasets, zero-shot evaluation".to_string(),
            priority: 1,
            category: DatasetCategory::General,
            languages: vec!["en".to_string()],
            domain: Some("Multiple (9 domains)".to_string()),
            queries: None,
            documents: None,
            access_method: AccessMethod::PythonFramework {
                package: "beir".to_string(),
                function: "util.download_dataset".to_string(),
            },
            format: DatasetFormat::Beir,
            url: Some("https://github.com/beir-cellar/beir".to_string()),
            citation: None,
            notes: Some("13 publicly available datasets".to_string()),
        });

        // Priority 2: High Value
        registry.add_dataset(DatasetEntry {
            name: "trec-dl-2023".to_string(),
            description: "TREC Deep Learning Track 2023".to_string(),
            priority: 2,
            category: DatasetCategory::General,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(200),
            documents: None,
            access_method: AccessMethod::DirectDownload {
                url: "https://trec.nist.gov/data/deep2023.html".to_string(),
            },
            format: DatasetFormat::Trec,
            url: Some("https://trec.nist.gov/".to_string()),
            citation: None,
            notes: Some("50+ runs per query, graded relevance".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "lotte".to_string(),
            description: "LoTTE - Long-tail topic-stratified evaluation".to_string(),
            priority: 2,
            category: DatasetCategory::General,
            languages: vec!["en".to_string()],
            domain: Some("Long-tail topics".to_string()),
            queries: Some(6_000),
            documents: Some(2_000_000),
            access_method: AccessMethod::HuggingFace {
                dataset_id: "mteb/LoTTE".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://huggingface.co/datasets/mteb/LoTTE".to_string()),
            citation: None,
            notes: Some("12 test sets, StackExchange topics".to_string()),
        });

        // Priority 3: Multilingual
        registry.add_dataset(DatasetEntry {
            name: "miracl".to_string(),
            description: "MIRACL - 18 languages, 40k+ queries".to_string(),
            priority: 3,
            category: DatasetCategory::Multilingual,
            languages: vec![
                "ar", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "it",
                "ja", "ko", "nl", "pl", "pt", "ru", "sw", "th", "zh",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            domain: None,
            queries: Some(40_203),
            documents: None,
            access_method: AccessMethod::HuggingFace {
                dataset_id: "mteb/miracl".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://project-miracl.github.io/".to_string()),
            citation: None,
            notes: Some("Human-annotated, Wikipedia-based".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "mteb".to_string(),
            description: "MTEB - 58 datasets, 112 languages".to_string(),
            priority: 3,
            category: DatasetCategory::Multilingual,
            languages: vec!["112 languages".to_string()],
            domain: Some("Multiple".to_string()),
            queries: None,
            documents: None,
            access_method: AccessMethod::PythonFramework {
                package: "mteb".to_string(),
                function: "MTEB".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: Some("mteb".to_string()),
            },
            url: Some("https://huggingface.co/mteb".to_string()),
            citation: None,
            notes: Some("Focus on retrieval and reranking tasks".to_string()),
        });

        // Priority 4: Domain-Specific
        registry.add_dataset(DatasetEntry {
            name: "legalbench-rag".to_string(),
            description: "LegalBench-RAG - Legal domain, precise retrieval".to_string(),
            priority: 4,
            category: DatasetCategory::DomainSpecific,
            languages: vec!["en".to_string()],
            domain: Some("Legal".to_string()),
            queries: Some(6_889),
            documents: Some(714),
            access_method: AccessMethod::DirectDownload {
                url: "https://github.com/HazyResearch/legalbench".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://arxiv.org/html/2408.10343v1".to_string()),
            citation: None,
            notes: Some("Precise snippet retrieval, 4 legal sub-domains".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "fiqa".to_string(),
            description: "FiQA - Financial domain, opinion-based QA".to_string(),
            priority: 4,
            category: DatasetCategory::DomainSpecific,
            languages: vec!["en".to_string()],
            domain: Some("Financial".to_string()),
            queries: None,
            documents: None,
            access_method: AccessMethod::HuggingFace {
                dataset_id: "LLukas22/fiqa".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://sites.google.com/view/fiqa/home".to_string()),
            citation: None,
            notes: Some("Aspect-based sentiment + opinion-based QA".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "bioasq".to_string(),
            description: "BioASQ - Biomedical domain".to_string(),
            priority: 4,
            category: DatasetCategory::DomainSpecific,
            languages: vec!["en".to_string()],
            domain: Some("Biomedical".to_string()),
            queries: None,
            documents: None,
            access_method: AccessMethod::DirectDownload {
                url: "https://bioasq.org".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://bioasq.org".to_string()),
            citation: None,
            notes: Some("Structured + unstructured, expert annotations".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "scifact-open".to_string(),
            description: "SciFact-Open - Scientific claim verification".to_string(),
            priority: 4,
            category: DatasetCategory::DomainSpecific,
            languages: vec!["en".to_string()],
            domain: Some("Scientific".to_string()),
            queries: None,
            documents: Some(500_000),
            access_method: AccessMethod::DirectDownload {
                url: "https://github.com/allenai/scifact".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://arxiv.org/abs/2210.13777".to_string()),
            citation: None,
            notes: Some("500k research abstracts, evidence retrieval".to_string()),
        });

        // Priority 5: Question Answering
        registry.add_dataset(DatasetEntry {
            name: "hotpotqa".to_string(),
            description: "HotpotQA - Multi-hop reasoning".to_string(),
            priority: 5,
            category: DatasetCategory::QuestionAnswering,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(112_779),
            documents: None,
            access_method: AccessMethod::HuggingFace {
                dataset_id: "hotpotqa/hotpotqa".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://hotpotqa.github.io".to_string()),
            citation: None,
            notes: Some("Multi-hop QA, supporting facts required".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "natural-questions".to_string(),
            description: "Natural Questions - Real Google queries".to_string(),
            priority: 5,
            category: DatasetCategory::QuestionAnswering,
            languages: vec!["en".to_string()],
            domain: None,
            queries: None,
            documents: None,
            access_method: AccessMethod::DirectDownload {
                url: "https://ai.google.com/research/NaturalQuestions/download".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://ai.google.com/research/NaturalQuestions".to_string()),
            citation: None,
            notes: Some("42GB, real user queries, Wikipedia".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "squad".to_string(),
            description: "SQuAD - Reading comprehension".to_string(),
            priority: 5,
            category: DatasetCategory::QuestionAnswering,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(107_785),
            documents: Some(536),
            access_method: AccessMethod::HuggingFace {
                dataset_id: "squad".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://rajpurkar.github.io/SQuAD-explorer/".to_string()),
            citation: None,
            notes: Some("SQuAD 2.0 includes unanswerable questions".to_string()),
        });

        // Priority 6: Regional
        registry.add_dataset(DatasetEntry {
            name: "fire".to_string(),
            description: "FIRE - South Asian languages".to_string(),
            priority: 6,
            category: DatasetCategory::Regional,
            languages: vec!["South Asian languages".to_string()],
            domain: None,
            queries: None,
            documents: None,
            access_method: AccessMethod::RegionalForum {
                website: "https://www.isical.ac.in/~fire/".to_string(),
            },
            format: DatasetFormat::Trec,
            url: Some("https://www.isical.ac.in/~fire/".to_string()),
            citation: None,
            notes: Some("Similar format to TREC".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "clef".to_string(),
            description: "CLEF - European languages, multimodal".to_string(),
            priority: 6,
            category: DatasetCategory::Regional,
            languages: vec!["Multiple European".to_string()],
            domain: None,
            queries: None,
            documents: None,
            access_method: AccessMethod::RegionalForum {
                website: "https://www.clef-initiative.eu".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://www.clef-initiative.eu".to_string()),
            citation: None,
            notes: Some("Multilingual, multimodal evaluation".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "ntcir".to_string(),
            description: "NTCIR - Asian languages, cross-lingual".to_string(),
            priority: 6,
            category: DatasetCategory::Regional,
            languages: vec!["Japanese and other Asian".to_string()],
            domain: None,
            queries: None,
            documents: None,
            access_method: AccessMethod::RegionalForum {
                website: "https://research.nii.ac.jp/ntcir/".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://research.nii.ac.jp/ntcir/".to_string()),
            citation: None,
            notes: Some("Since 1997, multiple tasks".to_string()),
        });

        // Priority 7: Specialized
        registry.add_dataset(DatasetEntry {
            name: "fultr".to_string(),
            description: "FULTR - Fusion learning, satisfaction-oriented".to_string(),
            priority: 7,
            category: DatasetCategory::Specialized,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(224_000_000),
            documents: Some(683_000_000),
            access_method: AccessMethod::RequiresAccess {
                contact: "See paper for access".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2025-fultr.pdf".to_string()),
            citation: None,
            notes: Some("224M queries, satisfaction-oriented labels".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "trec-covid".to_string(),
            description: "TREC-COVID - Biomedical crisis retrieval".to_string(),
            priority: 7,
            category: DatasetCategory::DomainSpecific,
            languages: vec!["en".to_string()],
            domain: Some("Biomedical".to_string()),
            queries: Some(50),
            documents: None,
            access_method: AccessMethod::PythonFramework {
                package: "ir_datasets".to_string(),
                function: "load('beir/trec-covid')".to_string(),
            },
            format: DatasetFormat::Trec,
            url: Some("https://ir.nist.gov/trec-covid/".to_string()),
            citation: None,
            notes: Some("Part of BEIR, high-quality judgments".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "ifir".to_string(),
            description: "IFIR - Instruction-following IR".to_string(),
            priority: 7,
            category: DatasetCategory::Specialized,
            languages: vec!["en".to_string()],
            domain: Some("Finance, Law, Healthcare, Scientific".to_string()),
            queries: Some(2_426),
            documents: None,
            access_method: AccessMethod::DirectDownload {
                url: "See paper for access".to_string(),
            },
            format: DatasetFormat::Custom {
                converter: None,
            },
            url: Some("https://aclanthology.org/2025.naacl-long.511.pdf".to_string()),
            citation: None,
            notes: Some("4 domains, 3 complexity levels".to_string()),
        });

        registry.add_dataset(DatasetEntry {
            name: "antique".to_string(),
            description: "ANTIQUE - Non-factoid questions".to_string(),
            priority: 7,
            category: DatasetCategory::QuestionAnswering,
            languages: vec!["en".to_string()],
            domain: None,
            queries: Some(2_626),
            documents: None,
            access_method: AccessMethod::HuggingFace {
                dataset_id: "antique".to_string(),
            },
            format: DatasetFormat::HuggingFace,
            url: Some("https://arxiv.org/abs/1905.08957".to_string()),
            citation: None,
            notes: Some("Yahoo! Answers, opinion-based".to_string()),
        });

        registry
    }

    fn add_dataset(&mut self, entry: DatasetEntry) {
        self.datasets.insert(entry.name.clone(), entry);
    }

    /// Get all datasets by priority.
    pub fn by_priority(&self, priority: u8) -> Vec<&DatasetEntry> {
        self.datasets
            .values()
            .filter(|d| d.priority == priority)
            .collect()
    }

    /// Get all datasets by category.
    pub fn by_category(&self, category: DatasetCategory) -> Vec<&DatasetEntry> {
        self.datasets
            .values()
            .filter(|d| d.category == category)
            .collect()
    }

    /// Get dataset by name.
    pub fn get(&self, name: &str) -> Option<&DatasetEntry> {
        self.datasets.get(name)
    }

    /// List all dataset names.
    pub fn list_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.datasets.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get all datasets.
    pub fn all(&self) -> Vec<&DatasetEntry> {
        let mut entries: Vec<&DatasetEntry> = self.datasets.values().collect();
        entries.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.name.cmp(&b.name))
        });
        entries
    }

    /// Save registry to JSON file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.datasets)?;
        std::fs::write(path.as_ref(), json)
            .with_context(|| format!("Failed to write registry: {:?}", path.as_ref()))?;
        Ok(())
    }

    /// Load registry from JSON file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read registry: {:?}", path.as_ref()))?;
        let datasets: HashMap<String, DatasetEntry> = serde_json::from_str(&content)?;
        Ok(DatasetRegistry { datasets })
    }
}

impl Default for DatasetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = DatasetRegistry::new();
        assert!(!registry.datasets.is_empty());
        assert!(registry.get("msmarco-passage").is_some());
        assert!(registry.get("beir").is_some());
    }

    #[test]
    fn test_by_priority() {
        let registry = DatasetRegistry::new();
        let priority_1 = registry.by_priority(1);
        assert!(!priority_1.is_empty());
    }

    #[test]
    fn test_by_category() {
        let registry = DatasetRegistry::new();
        let multilingual = registry.by_category(DatasetCategory::Multilingual);
        assert!(!multilingual.is_empty());
    }
}

