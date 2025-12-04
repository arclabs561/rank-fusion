# Evaluation Dataset Recommendations for Rank Fusion

This document provides research-backed recommendations for evaluation datasets that would be most valuable for the `rank-fusion` repository, considering all implemented fusion methods and potential future implementations.

## Executive Summary

**Priority 1 (Start Here):**
- **MS MARCO Passage Ranking** - Industry standard, large-scale, multiple retrieval runs available
- **BEIR (13 public datasets)** - Zero-shot generalization testing, diverse domains

**Priority 2 (High Value):**
- **TREC Deep Learning Track runs** - Community-validated runs, multiple fusion scenarios
- **LoTTE** - Long-tail queries, specialized topics, complements mainstream benchmarks

**Priority 3 (Specialized Use Cases):**
- **FULTR** - Fusion learning signals, satisfaction-oriented ranking (if accessible)
- **TREC-COVID** - Domain-specific (biomedical), crisis information retrieval

## Priority 1: Essential Datasets

### MS MARCO Passage Ranking

**Why it's essential:**
- **Industry standard**: Most widely used benchmark in IR research
- **Large scale**: 367,013 training queries, 124,000 test queries
- **Multiple retrieval runs available**: BM25, dense embeddings, hybrid approaches
- **Establishes baseline**: Enables comparison with published results

**What it tests:**
- Score distribution differences (BM25 vs dense embeddings)
- Standardized fusion effectiveness (ERANK shows 2-5% NDCG improvement)
- Additive multi-task fusion for different retrieval signals
- Generalization from training to test sets

**Implementation notes:**
- Use MS MARCO v2 datasets (cleaner, less biased than v1)
- Focus on passage ranking task (more queries than document ranking)
- Available via `ir_datasets` Python library or direct download
- TREC format runs available from multiple participants

**Expected insights:**
- Validate that standardized fusion outperforms CombSUM when score distributions differ
- Confirm RRF effectiveness when score scales are incompatible
- Test weighted fusion when retriever quality is known a priori

**Dataset characteristics:**
- **Queries**: 124,000 test queries (held out from training)
- **Documents**: 8.8M passages
- **Relevance**: Sparse (typically 1 positive per query in training)
- **Format**: TREC runs available, qrels available

### BEIR (Benchmark IR)

**Why it's essential:**
- **Zero-shot evaluation**: Tests generalization across domains
- **Diverse domains**: 18 datasets across 9 domains (13 publicly available)
- **Reveals generalization gaps**: Shows in-domain performance doesn't predict out-of-domain
- **BM25 baseline**: Strong baseline for comparison

**What it tests:**
- Cross-domain generalization of fusion methods
- Robustness to different query types and document styles
- Whether fusion methods maintain effectiveness across domains
- Long-tail knowledge retrieval (complementary to LoTTE)

**Key BEIR datasets to prioritize:**
1. **Natural Questions (NQ)** - 4,352 queries, binary relevance
2. **HotpotQA** - Multi-hop reasoning, 7,405 queries
3. **FEVER** - Fact verification, 6,666 queries
4. **ArguAna** - Argument retrieval, 1,406 queries
5. **SciFact** - Scientific fact checking, 300 queries
6. **TREC-COVID** - Biomedical, 50 queries (high-quality judgments)
7. **NFCorpus** - Medical information, 3,233 queries
8. **Quora** - Duplicate question detection, 10,000 queries
9. **SCIDOCS** - Scientific document similarity, 25,657 queries
10. **Signal-1M** - News articles, 97 queries
11. **TREC-NEWS** - News articles, 57 queries
12. **Robust04** - Ad-hoc retrieval, 250 queries (if accessible)
13. **CQADupStack** - StackExchange questions, 13,145 queries

**Implementation notes:**
- Use `ir_datasets` library for easy access
- Focus on 13 publicly available datasets
- Report nDCG@10 per dataset and aggregate average
- Note that BM25 often outperforms dense models on BEIR

**Expected insights:**
- Identify which fusion methods generalize best across domains
- Discover domain-specific fusion method preferences
- Validate that standardized fusion maintains improvements out-of-domain

**Critical finding from research:**
> "In-domain performance is not a good indicator for out-of-domain generalization. Dense embedding models trained on MS MARCO underperform BM25 baseline across almost all BEIR datasets."

This suggests fusion methods that work well on MS MARCO may need different configurations for BEIR.

## Priority 2: High-Value Datasets

### TREC Deep Learning Track Runs

**Why it's valuable:**
- **Community-validated runs**: Multiple participants, diverse approaches
- **Multiple runs per query**: Perfect for fusion evaluation
- **Rigorous evaluation**: Human relevance judgments, graded relevance
- **Recent data**: 2023 track uses MS MARCO v2, synthetic queries

**What it tests:**
- Fusion of multiple retrieval systems (BM25, dense, hybrid)
- Effectiveness when combining complementary retrieval approaches
- Real-world fusion scenarios with diverse system outputs

**Implementation notes:**
- TREC 2023 Deep Learning track has 50+ runs per query
- Runs available in standard TREC format
- Qrels available with graded relevance (0-4 scale)
- Can fuse top-performing runs to test fusion effectiveness

**Expected insights:**
- Which fusion methods work best when combining multiple strong systems
- Whether fusion can improve upon best individual systems
- Optimal fusion configurations for multi-system scenarios

**Dataset characteristics:**
- **Queries**: 200 test queries (2023 track)
- **Runs**: 50+ runs per query from different participants
- **Relevance**: Graded (0-4), high-quality human judgments
- **Format**: Standard TREC run format, qrels format

### LoTTE (Long-Tail Topic-stratified Evaluation)

**Why it's valuable:**
- **Long-tail focus**: Specialized topics underrepresented in general benchmarks
- **Real-world queries**: Natural information-seeking queries from StackExchange
- **Topic stratification**: Enables domain-specific analysis
- **Complementary to BEIR**: Tests different aspect of generalization

**What it tests:**
- Fusion effectiveness on specialized, niche topics
- Whether fusion methods help with domain-specific terminology
- Performance on queries requiring domain expertise
- Long-tail knowledge retrieval scenarios

**Implementation notes:**
- 12 test sets, each 500-2,000 queries
- Topics from StackExchange communities (technology, lifestyle, professional)
- 100k-2M passages per test set
- Available via HuggingFace datasets

**Expected insights:**
- Identify fusion methods that excel on specialized topics
- Understand whether fusion helps with vocabulary mismatch in niche domains
- Test if standardized fusion handles domain-specific score distributions

**Dataset characteristics:**
- **Queries**: 500-2,000 per test set (12 test sets total)
- **Documents**: 100k-2M passages per test set
- **Topics**: StackExchange communities (diverse domains)
- **Format**: Available via HuggingFace datasets

## Priority 3: Multilingual and Cross-Lingual Datasets

### MIRACL (Multilingual Information Retrieval Across a Continuum of Languages)

**Why it's valuable:**
- **18 languages**: Comprehensive multilingual evaluation
- **Human-annotated**: Thorough relevance judgments across languages
- **Large scale**: 40,203 queries, 343,177 judgment pairs
- **Wikipedia-based**: Real-world knowledge base content

**What it tests:**
- Cross-lingual fusion effectiveness
- Multilingual retrieval system fusion
- Language-specific fusion method preferences
- Generalization across language families

**Implementation notes:**
- Available via HuggingFace Datasets
- Baseline results available (BM25, mDPR)
- Passage-level retrieval (similar to MS MARCO)
- Topics and judgments available through HuggingFace

**Expected insights:**
- Which fusion methods work best across languages
- Whether fusion helps with cross-lingual retrieval
- Language-specific fusion configuration needs

**Dataset characteristics:**
- **Queries**: 40,203 total (training + dev)
- **Languages**: 18 diverse languages
- **Documents**: Wikipedia passages
- **Format**: Available via HuggingFace, can convert to TREC format

### MTEB (Massive Text Embedding Benchmark)

**Why it's valuable:**
- **Comprehensive**: 58 datasets covering 112 languages
- **8 task categories**: Classification, clustering, retrieval, reranking, STS, etc.
- **Standard leaderboard**: Widely used for embedding evaluation
- **Diverse domains**: Legal, code, healthcare, and more

**What it tests:**
- Fusion of embedding-based retrievers
- Cross-task fusion effectiveness
- Domain-specific fusion (legal, healthcare, code)
- Multilingual embedding fusion

**Implementation notes:**
- Python framework available
- Can extract retrieval tasks for fusion evaluation
- Multiple domains enable domain-specific analysis
- Standard evaluation protocol

**Expected insights:**
- How fusion performs across different embedding tasks
- Domain-specific fusion method preferences
- Multilingual embedding fusion strategies

**Dataset characteristics:**
- **Datasets**: 58 total across 8 categories
- **Languages**: 112 languages
- **Tasks**: Retrieval, reranking, classification, clustering, etc.
- **Format**: Python framework, can export to TREC format

## Priority 4: Domain-Specific Specialized Datasets

### LegalBench-RAG

**Why it's valuable:**
- **Legal domain**: First RAG benchmark for legal retrieval
- **Precise retrieval**: Focuses on minimal, highly relevant text segments
- **Multiple datasets**: PrivacyQA, CUAD, MAUD, ContractNLI
- **Large scale**: 80M characters, 714 documents, 6,889 Q&A pairs

**What it tests:**
- Fusion for precise legal document retrieval
- Domain-specific fusion effectiveness
- Fusion when precision matters more than recall
- Specialized terminology handling

**Implementation notes:**
- Available from LegalBench project
- Requires conversion to TREC format
- Focus on precise snippet retrieval
- Multiple legal sub-domains

**Expected insights:**
- Whether fusion improves precision in legal retrieval
- Optimal fusion methods for domain-specific retrieval
- How fusion handles specialized terminology

**Dataset characteristics:**
- **Documents**: 714 legal documents
- **Queries**: 6,889 question-answer pairs
- **Domain**: Legal (privacy, contracts, M&A, NLI)
- **Format**: May require conversion to TREC format

### FiQA (Financial Information Retrieval)

**Why it's valuable:**
- **Financial domain**: Specialized financial language and concepts
- **Aspect-based**: Sentiment analysis and opinion-based QA
- **Real-world queries**: Financial information needs
- **Dual tasks**: Sentiment analysis + question answering

**What it tests:**
- Fusion for financial domain retrieval
- Opinion-based retrieval fusion
- Specialized domain language handling
- Multi-task fusion (sentiment + retrieval)

**Implementation notes:**
- Available via HuggingFace or FiQA challenge website
- Combines structured and unstructured financial data
- Evaluation metrics: Precision, Recall, F-score, NDCG, MRR
- Can convert to TREC format

**Expected insights:**
- Domain-specific fusion method effectiveness
- How fusion handles opinionated content
- Financial terminology fusion strategies

**Dataset characteristics:**
- **Domain**: Financial (microblogs, reports, news)
- **Tasks**: Aspect-based sentiment + opinion-based QA
- **Format**: Available via HuggingFace, can convert to TREC

### BioASQ

**Why it's valuable:**
- **Biomedical domain**: Comprehensive biomedical IR benchmark
- **Structured + unstructured**: Combines documents and ontologies
- **Expert annotations**: Biomedical expert quality assurance
- **Multiple tasks**: Semantic indexing + question answering

**What it tests:**
- Fusion for biomedical information retrieval
- Domain-specific fusion effectiveness
- Fusion with structured knowledge (ontologies)
- Expert-level information needs

**Implementation notes:**
- Available from BioASQ website
- Annual challenges with updated datasets
- Combines documents and biomedical ontologies
- Requires conversion to TREC format

**Expected insights:**
- Biomedical domain fusion effectiveness
- How fusion integrates structured and unstructured data
- Expert-level query fusion strategies

**Dataset characteristics:**
- **Domain**: Biomedical
- **Data types**: Documents + biomedical ontologies
- **Tasks**: Semantic indexing, question answering
- **Format**: Available from BioASQ, requires conversion

### SciFact-Open

**Why it's valuable:**
- **Scientific domain**: Scientific claim verification
- **Large scale**: 500,000 research abstracts
- **Open-domain**: Realistic scientific retrieval scenario
- **Claim verification**: Tests evidence retrieval

**What it tests:**
- Fusion for scientific literature retrieval
- Large-scale scientific corpus fusion
- Evidence retrieval fusion
- Open-domain scientific retrieval

**Implementation notes:**
- Available from SciFact project
- Large corpus (500k abstracts)
- Claim verification task
- Requires conversion to TREC format

**Expected insights:**
- Scientific domain fusion effectiveness
- Large-scale corpus fusion strategies
- Evidence retrieval fusion methods

**Dataset characteristics:**
- **Documents**: 500,000 research abstracts
- **Task**: Scientific claim verification
- **Domain**: Scientific literature
- **Format**: Available from project, requires conversion

## Priority 5: Question Answering and Complex Retrieval

### HotpotQA

**Why it's valuable:**
- **Multi-hop reasoning**: Requires information from multiple documents
- **Large scale**: 112,779 question-answer pairs
- **Explainability**: Supporting facts required
- **Two settings**: Distractor and full wiki

**What it tests:**
- Fusion for multi-hop retrieval
- Multi-document fusion effectiveness
- Explainable retrieval fusion
- Complex reasoning fusion

**Implementation notes:**
- Available via HuggingFace or HotpotQA website
- Two evaluation settings (distractor vs full wiki)
- Requires supporting fact identification
- Can convert to TREC format

**Expected insights:**
- How fusion helps with multi-hop retrieval
- Multi-document fusion strategies
- Explainable fusion approaches

**Dataset characteristics:**
- **Queries**: 112,779 question-answer pairs
- **Task**: Multi-hop question answering
- **Documents**: Wikipedia
- **Format**: Available via HuggingFace, can convert to TREC

### Natural Questions (NQ)

**Why it's valuable:**
- **Real queries**: From actual Google searches
- **Large scale**: 42GB of data
- **Wikipedia-based**: High-quality knowledge base
- **Answer spans**: Precise answer location

**What it tests:**
- Fusion for real-world query patterns
- Large-scale retrieval fusion
- Answer span retrieval fusion
- User query fusion effectiveness

**Implementation notes:**
- Available from Google Research
- Large dataset (42GB)
- Real user queries
- Can extract retrieval task for fusion

**Expected insights:**
- Real-world query fusion effectiveness
- Large-scale fusion performance
- User query pattern fusion

**Dataset characteristics:**
- **Queries**: Real Google search queries
- **Documents**: Wikipedia
- **Size**: 42GB total
- **Format**: Available from Google, requires processing

### SQuAD (Stanford Question Answering Dataset)

**Why it's valuable:**
- **Reading comprehension**: Passage-level QA
- **Large scale**: 107,785+ question-answer pairs
- **Answer spans**: Precise text span identification
- **SQuAD 2.0**: Includes unanswerable questions

**What it tests:**
- Fusion for reading comprehension
- Passage-level retrieval fusion
- Answer span fusion
- Unanswerable question handling

**Implementation notes:**
- Available via HuggingFace or SQuAD website
- SQuAD 2.0 includes unanswerable questions
- Passage-level retrieval task
- Can convert to TREC format

**Expected insights:**
- Passage-level fusion effectiveness
- Answer span fusion strategies
- Unanswerable question fusion handling

**Dataset characteristics:**
- **Queries**: 107,785+ question-answer pairs
- **Documents**: Wikipedia articles (536 articles)
- **Task**: Reading comprehension
- **Format**: Available via HuggingFace

## Priority 6: Regional and Language-Specific Datasets

### FIRE (Forum for Information Retrieval Evaluation)

**Why it's valuable:**
- **South Asian languages**: Focus on regional languages
- **Community-driven**: Similar to TREC/CLEF model
- **Reusable collections**: Standard test collections
- **Regional context**: Reflects regional information needs

**What it tests:**
- Fusion for South Asian languages
- Regional information need fusion
- Language-specific fusion effectiveness
- Community-driven evaluation

**Implementation notes:**
- Available from FIRE website
- Multiple languages and tasks
- Similar format to TREC
- Can convert to TREC format

**Expected insights:**
- Regional language fusion effectiveness
- Community-driven evaluation fusion
- Language-specific fusion strategies

**Dataset characteristics:**
- **Languages**: South Asian languages
- **Format**: Similar to TREC
- **Source**: FIRE website
- **Tasks**: Multiple IR tasks

### CLEF (Conference and Labs of the Evaluation Forum)

**Why it's valuable:**
- **Multilingual focus**: European and non-European languages
- **Multiple labs**: Specialized evaluation campaigns
- **Multimodal**: Beyond text to multimodal retrieval
- **Long-running**: Established evaluation tradition

**What it tests:**
- Multilingual fusion effectiveness
- Cross-lingual fusion
- Multimodal fusion (if applicable)
- European language fusion

**Implementation notes:**
- Available from CLEF website
- Multiple labs and tasks
- Multilingual collections
- Can convert to TREC format

**Expected insights:**
- Multilingual fusion strategies
- Cross-lingual fusion effectiveness
- European language fusion methods

**Dataset characteristics:**
- **Languages**: Multiple European and non-European
- **Tasks**: Various IR tasks
- **Format**: CLEF format, can convert to TREC
- **Source**: CLEF website

### NTCIR (NII Test Collection for IR Systems)

**Why it's valuable:**
- **Asian languages**: Japanese and other Asian languages
- **Cross-lingual**: Cross-lingual IR evaluation
- **Long-running**: Since 1997, established tradition
- **Multiple tasks**: IR, QA, summarization, extraction

**What it tests:**
- Asian language fusion effectiveness
- Cross-lingual fusion
- Japanese language fusion
- Multi-task fusion

**Implementation notes:**
- Available from NTCIR website
- Multiple editions and tasks
- Asian language focus
- Can convert to TREC format

**Expected insights:**
- Asian language fusion strategies
- Cross-lingual fusion effectiveness
- Japanese language fusion methods

**Dataset characteristics:**
- **Languages**: Japanese and other Asian languages
- **Tasks**: IR, QA, summarization, extraction
- **Format**: NTCIR format, can convert to TREC
- **Source**: NTCIR website

## Priority 7: Specialized Use Cases

### FULTR (Fusion Learning to Rank)

**Why it's valuable:**
- **Fusion-specific**: Designed specifically for fusion learning research
- **Real-world signals**: 224M queries, 683M documents from Baidu Search
- **Satisfaction-oriented**: Combines relevance and user satisfaction signals
- **Heterogeneous signals**: Prior attributes (relevance) + posterior attributes (clicks)

**What it tests:**
- Fusion of heterogeneous information sources
- Satisfaction-oriented ranking (beyond pure relevance)
- Integration of prior and posterior signals
- Real-world robustness at scale

**Implementation notes:**
- May require access permissions or data sharing agreements
- Large scale requires efficient implementation
- Two subsets: prior-attribute (54k queries) and posterior-attribute (larger)
- Satisfaction labels: {0-bad, 1-fair, 2-good, 3-excellent, 4-perfect}

**Expected insights:**
- How to fuse relevance signals with user behavior signals
- Whether additive multi-task fusion works for satisfaction-oriented ranking
- Optimal fusion strategies for heterogeneous information integration

**Dataset characteristics:**
- **Queries**: 224M total (prior: 54k, posterior: larger)
- **Documents**: 683M total
- **Labels**: Satisfaction-oriented (0-4 scale)
- **Format**: May require special access

### TREC-COVID

**Why it's valuable:**
- **Domain-specific**: Biomedical information retrieval
- **Crisis scenario**: Tests fusion under time-sensitive information needs
- **High-quality judgments**: 50 queries, 493.5 judgments per query (average)
- **Graded relevance**: Multiple relevance levels

**What it tests:**
- Fusion effectiveness in specialized domains
- Performance on domain-specific terminology and concepts
- Crisis information retrieval scenarios
- Fusion when one retriever may be domain-specialized

**Implementation notes:**
- Part of BEIR benchmark (already included if using BEIR)
- Can be used standalone for domain-specific evaluation
- High-quality judgments enable reliable evaluation
- Available via `ir_datasets`

**Expected insights:**
- Whether fusion methods need domain-specific tuning
- How fusion performs when combining general and domain-specific retrievers
- Effectiveness of weighted fusion when domain expertise is known

**Dataset characteristics:**
- **Queries**: 50 test queries
- **Documents**: CORD-19 biomedical literature
- **Relevance**: Graded, high-quality (493.5 judgments/query average)
- **Format**: Available via ir_datasets, part of BEIR

### IFIR (Instruction-Following Information Retrieval)

**Why it's valuable:**
- **Complex queries**: Instruction-following rather than simple keywords
- **Specialized domains**: Finance, law, healthcare, scientific literature
- **Complexity levels**: Three levels of query complexity
- **Real-world scenarios**: Reflects actual professional information needs

**What it tests:**
- Fusion for complex, instruction-based queries
- Domain-specific instruction fusion
- Multi-criteria retrieval fusion
- Professional information need fusion

**Implementation notes:**
- Available from IFIR project
- 2,426 examples across 8 subsets
- 4 domains × 2 complexity levels
- Requires conversion to TREC format

**Expected insights:**
- Complex query fusion effectiveness
- Instruction-following fusion strategies
- Domain-specific instruction fusion

**Dataset characteristics:**
- **Queries**: 2,426 examples
- **Domains**: Finance, law, healthcare, scientific
- **Complexity**: Three levels
- **Format**: Available from project, requires conversion

### ANTIQUE (Answering Non-factoid Questions)

**Why it's valuable:**
- **Non-factoid**: Subjective, opinion-based questions
- **Real queries**: From Yahoo! Answers community
- **Diverse categories**: Multiple question types
- **Relevance annotations**: 34,011 manual annotations

**What it tests:**
- Fusion for non-factoid retrieval
- Opinion-based retrieval fusion
- Subjective information fusion
- Community question fusion

**Implementation notes:**
- Available via HuggingFace or ANTIQUE website
- Non-factoid questions (2,626 queries)
- Manual relevance annotations
- Can convert to TREC format

**Expected insights:**
- Non-factoid fusion effectiveness
- Opinion-based fusion strategies
- Subjective information fusion methods

**Dataset characteristics:**
- **Queries**: 2,626 non-factoid questions
- **Source**: Yahoo! Answers
- **Annotations**: 34,011 manual relevance judgments
- **Format**: Available via HuggingFace

### BordIRlines (Cross-Lingual Geopolitical Bias)

**Why it's valuable:**
- **Geopolitical bias**: Tests bias in cross-lingual retrieval
- **Multilingual**: Multiple languages and perspectives
- **RAG evaluation**: Specifically for RAG systems
- **Bias detection**: Identifies retrieval bias

**What it tests:**
- Cross-lingual fusion effectiveness
- Bias-aware fusion
- Multilingual perspective fusion
- Geopolitical information fusion

**Implementation notes:**
- Available from BordIRlines project
- Multilingual Wikipedia articles
- Geopolitical questions
- Requires conversion to TREC format

**Expected insights:**
- Cross-lingual fusion strategies
- Bias-aware fusion methods
- Multilingual perspective fusion

**Dataset characteristics:**
- **Domain**: Geopolitical information
- **Languages**: Multiple languages
- **Task**: Cross-lingual RAG evaluation
- **Format**: Available from project

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **MS MARCO Passage Ranking**
   - Download MS MARCO v2 passage dataset
   - Load BM25 and dense retrieval runs
   - Implement evaluation pipeline using existing `real_world.rs`
   - Compare all fusion methods (RRF, CombSUM, standardized, etc.)
   - Generate baseline results

2. **BEIR - Core Datasets**
   - Start with 3-5 BEIR datasets (NQ, HotpotQA, FEVER)
   - Evaluate fusion methods on each
   - Compare in-domain (MS MARCO) vs out-of-domain (BEIR) performance
   - Identify generalization patterns

### Phase 2: Expansion (Weeks 3-4)
3. **TREC Deep Learning Track**
   - Download TREC 2023 runs
   - Implement multi-run fusion (3+ runs per query)
   - Test fusion of complementary retrieval systems
   - Compare fusion vs best individual systems

4. **LoTTE**
   - Load LoTTE test sets
   - Evaluate on long-tail topics
   - Analyze domain-specific performance
   - Compare with BEIR results

5. **MIRACL (Multilingual)**
   - Load MIRACL datasets for 2-3 languages
   - Evaluate fusion across languages
   - Compare multilingual fusion effectiveness
   - Identify language-specific patterns

### Phase 3: Domain Specialization (Weeks 5-6)
6. **Domain-Specific Datasets**
   - LegalBench-RAG (legal domain)
   - FiQA (financial domain)
   - BioASQ or SciFact-Open (biomedical/scientific)
   - Evaluate domain-specific fusion effectiveness

7. **Question Answering Datasets**
   - HotpotQA (multi-hop)
   - Natural Questions (real queries)
   - SQuAD (reading comprehension)
   - Test fusion for QA retrieval

### Phase 4: Advanced (Weeks 7-8)
8. **Additional BEIR datasets**
   - Complete evaluation on all 13 public BEIR datasets
   - Generate comprehensive cross-domain analysis
   - Identify dataset-specific fusion preferences

9. **MTEB Retrieval Tasks**
   - Extract retrieval tasks from MTEB
   - Evaluate fusion across MTEB domains
   - Compare with embedding-based fusion

10. **Regional Datasets** (if accessible)
    - FIRE (South Asian languages)
    - CLEF (European languages)
    - NTCIR (Asian languages)
    - Evaluate regional language fusion

### Phase 5: Specialized (Weeks 9-10)
11. **FULTR** (if accessible)
    - Evaluate satisfaction-oriented fusion
    - Test heterogeneous signal integration
    - Compare prior vs posterior attribute fusion

12. **Complex Retrieval**
    - IFIR (instruction-following)
    - ANTIQUE (non-factoid)
    - BordIRlines (cross-lingual bias)
    - Test advanced fusion scenarios

## Evaluation Metrics to Report

For each dataset, report:
- **nDCG@10** - Primary metric (handles graded relevance)
- **nDCG@100** - Deeper ranking evaluation
- **MAP** - Mean Average Precision (binary/graded)
- **MRR** - Mean Reciprocal Rank (first relevant result)
- **Precision@10** - Top-10 precision
- **Recall@100** - Coverage of relevant documents

## Key Research Questions to Answer

1. **Does standardized fusion maintain 2-5% NDCG improvement on real datasets?**
   - Test on MS MARCO (in-domain) and BEIR (out-of-domain)

2. **Which fusion methods generalize best across domains?**
   - Compare RRF, CombSUM, standardized, additive multi-task on BEIR

3. **Does fusion improve upon best individual systems?**
   - Test on TREC runs where individual systems are already strong

4. **How do fusion methods perform on long-tail queries?**
   - Evaluate on LoTTE specialized topics

5. **What fusion configurations work best for different scenarios?**
   - Score distribution differences → standardized
   - Known retriever quality → weighted
   - E-commerce multi-task → additive multi-task
   - Incompatible scales → RRF

## Dataset Access and Tools

### Recommended Tools
- **ir_datasets**: Python library for accessing IR datasets
  ```python
  import ir_datasets
  dataset = ir_datasets.load("msmarco-passage/train")
  ```
- **BEIR**: Python framework for BEIR evaluation
  ```python
  from beir import util, LoggingHandler
  ```
- **TREC tools**: Standard TREC format parsers (already implemented in `real_world.rs`)

### Data Sources
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **BEIR**: https://github.com/beir-cellar/beir
- **TREC**: https://trec.nist.gov/data/
- **LoTTE**: https://huggingface.co/datasets/mteb/LoTTE
- **ir_datasets**: https://ir-datasets.com/

## Expected Outcomes

### Validation Goals
1. **Confirm research claims**: Validate that standardized fusion shows 2-5% improvement when score distributions differ
2. **Identify best practices**: Determine optimal fusion methods for different scenarios
3. **Generalization analysis**: Understand which methods work across domains vs domain-specific
4. **Performance benchmarks**: Establish baseline performance for all fusion methods

### Deliverables
1. **Evaluation reports**: HTML reports (like existing synthetic scenario reports) for each dataset
2. **Comparison tables**: Cross-dataset comparison of fusion methods
3. **Best practices guide**: Recommendations for which fusion method to use when
4. **Performance benchmarks**: Timing and accuracy benchmarks for all methods

## Notes on Implementation

The existing `evals/src/real_world.rs` module already provides:
- TREC run file loading
- Qrels loading
- Metrics computation (nDCG, MAP, MRR, Precision, Recall)
- Evaluation framework for standardized fusion

**Next steps:**
1. Extend `real_world.rs` to support all fusion methods (not just standardized)
2. Add dataset loaders for MS MARCO, BEIR, LoTTE
3. Create evaluation pipeline that runs all methods on all datasets
4. Generate comprehensive HTML reports comparing methods across datasets

## Complete Dataset List

### Priority 1: Essential (Start Here)
1. MS MARCO Passage Ranking
2. BEIR (13 public datasets)

### Priority 2: High Value
3. TREC Deep Learning Track
4. LoTTE

### Priority 3: Multilingual
5. MIRACL (18 languages)
6. MTEB (58 datasets, 112 languages)

### Priority 4: Domain-Specific
7. LegalBench-RAG
8. FiQA (Financial)
9. BioASQ (Biomedical)
10. SciFact-Open (Scientific)

### Priority 5: Question Answering
11. HotpotQA (Multi-hop)
12. Natural Questions
13. SQuAD (Reading comprehension)

### Priority 6: Regional/Language-Specific
14. FIRE (South Asian languages)
15. CLEF (European languages)
16. NTCIR (Asian languages)

### Priority 7: Specialized
17. FULTR (Fusion learning)
18. TREC-COVID (Biomedical crisis)
19. IFIR (Instruction-following)
20. ANTIQUE (Non-factoid)
21. BordIRlines (Cross-lingual bias)

## Dataset Access Summary

| Dataset | Access Method | Format | Languages |
|---------|--------------|--------|-----------|
| MS MARCO | Website / ir_datasets | TREC | English |
| BEIR | GitHub / ir_datasets | TREC | English |
| MIRACL | HuggingFace | HuggingFace | 18 languages |
| MTEB | HuggingFace / Python | Python | 112 languages |
| LoTTE | HuggingFace | HuggingFace | English |
| LegalBench-RAG | Project website | Custom | English |
| FiQA | HuggingFace | HuggingFace | English |
| BioASQ | BioASQ website | Custom | English |
| HotpotQA | HuggingFace | HuggingFace | English |
| Natural Questions | Google Research | Custom | English |
| SQuAD | HuggingFace | HuggingFace | English |
| FIRE | FIRE website | TREC-like | South Asian |
| CLEF | CLEF website | CLEF format | Multiple |
| NTCIR | NTCIR website | NTCIR format | Asian |
| TREC-COVID | ir_datasets | TREC | English |

## References

- MS MARCO: https://microsoft.github.io/msmarco/
- BEIR: https://github.com/beir-cellar/beir
- TREC 2023 Deep Learning: https://trec.nist.gov/pubs/trec32/papers/Overview_deep.pdf
- LoTTE: https://arxiv.org/pdf/2112.01488.pdf
- MIRACL: https://project-miracl.github.io/
- MTEB: https://huggingface.co/mteb
- LegalBench-RAG: https://arxiv.org/html/2408.10343v1
- FiQA: https://sites.google.com/view/fiqa/home
- BioASQ: https://bioasq.org
- SciFact-Open: https://arxiv.org/abs/2210.13777
- HotpotQA: https://hotpotqa.github.io
- Natural Questions: https://ai.google.com/research/NaturalQuestions/download
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- FIRE: https://www.isical.ac.in/~fire/
- CLEF: https://www.clef-initiative.eu
- NTCIR: https://research.nii.ac.jp/ntcir/
- FULTR: https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/li-2025-fultr.pdf
- IFIR: https://aclanthology.org/2025.naacl-long.511.pdf
- ANTIQUE: https://arxiv.org/abs/1905.08957
- BordIRlines: https://arxiv.org/html/2410.01171v1
- ir_datasets: https://ir-datasets.com/

