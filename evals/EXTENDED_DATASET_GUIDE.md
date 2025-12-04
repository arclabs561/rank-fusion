# Extended Dataset Guide: 21+ Evaluation Datasets

This guide expands on `DATASET_RECOMMENDATIONS.md` with additional datasets discovered through research.

## Quick Reference: All Datasets

### Essential (Priority 1)
1. **MS MARCO** - Industry standard, large-scale
2. **BEIR** - 13 public datasets, zero-shot evaluation

### High Value (Priority 2)
3. **TREC Deep Learning** - Community-validated runs
4. **LoTTE** - Long-tail topics, specialized queries

### Multilingual (Priority 3)
5. **MIRACL** - 18 languages, 40k+ queries
6. **MTEB** - 58 datasets, 112 languages, 8 task categories

### Domain-Specific (Priority 4)
7. **LegalBench-RAG** - Legal domain, precise retrieval
8. **FiQA** - Financial domain, opinion-based QA
9. **BioASQ** - Biomedical domain, structured + unstructured
10. **SciFact-Open** - Scientific claim verification, 500k abstracts

### Question Answering (Priority 5)
11. **HotpotQA** - Multi-hop reasoning, 112k+ queries
12. **Natural Questions** - Real Google queries, 42GB
13. **SQuAD** - Reading comprehension, 107k+ queries

### Regional/Language-Specific (Priority 6)
14. **FIRE** - South Asian languages
15. **CLEF** - European languages, multimodal
16. **NTCIR** - Asian languages, cross-lingual

### Specialized (Priority 7)
17. **FULTR** - Fusion learning, satisfaction-oriented
18. **TREC-COVID** - Biomedical crisis retrieval
19. **IFIR** - Instruction-following IR, 4 domains
20. **ANTIQUE** - Non-factoid questions, 2.6k queries
21. **BordIRlines** - Cross-lingual geopolitical bias

## Dataset Characteristics Matrix

| Dataset | Queries | Documents | Languages | Domain | Format |
|---------|---------|-----------|-----------|--------|--------|
| MS MARCO | 124k | 8.8M | English | General | TREC |
| BEIR | Varies | Varies | English | 9 domains | TREC |
| MIRACL | 40k | Varies | 18 | General | HuggingFace |
| MTEB | Varies | Varies | 112 | Multiple | Python/HF |
| LoTTE | 6k-24k | 100k-2M | English | Long-tail | HuggingFace |
| LegalBench-RAG | 6.9k | 714 | English | Legal | Custom |
| FiQA | Varies | Varies | English | Financial | HuggingFace |
| BioASQ | Varies | Varies | English | Biomedical | Custom |
| HotpotQA | 112k | Wikipedia | English | Multi-hop | HuggingFace |
| Natural Questions | Large | Wikipedia | English | General | Custom |
| SQuAD | 107k | 536 articles | English | Reading | HuggingFace |

## Access Methods

### HuggingFace Datasets (Easiest)
- MIRACL: `datasets.load_dataset("mteb/miracl")`
- LoTTE: `datasets.load_dataset("mteb/LoTTE")`
- HotpotQA: `datasets.load_dataset("hotpotqa/hotpotqa")`
- SQuAD: `datasets.load_dataset("squad")`
- FiQA: `datasets.load_dataset("LLukas22/fiqa")`

### Python Frameworks
- MTEB: `pip install mteb`
- BEIR: `pip install beir`
- ir_datasets: `pip install ir_datasets`

### Direct Download
- MS MARCO: https://microsoft.github.io/msmarco/
- TREC: https://trec.nist.gov/
- LegalBench-RAG: Project website
- BioASQ: https://bioasq.org
- Natural Questions: Google Research

### Regional Forums
- FIRE: https://www.isical.ac.in/~fire/
- CLEF: https://www.clef-initiative.eu
- NTCIR: https://research.nii.ac.jp/ntcir/

## Format Conversion Guide

Most datasets need conversion to TREC format for our evaluation system:

### From HuggingFace to TREC

```python
from datasets import load_dataset
import json

# Load dataset
ds = load_dataset("mteb/miracl", "en", split="dev")

# Convert to TREC runs format
with open("runs.txt", "w") as f:
    for example in ds:
        query_id = example["query_id"]
        # ... process and write in TREC format
        f.write(f"{query_id} Q0 {doc_id} {rank} {score} run_tag\n")

# Convert to TREC qrels format
with open("qrels.txt", "w") as f:
    for example in ds:
        query_id = example["query_id"]
        for rel in example["positive_passages"]:
            f.write(f"{query_id} 0 {rel['docid']} 1\n")
```

### From MTEB to TREC

```python
from mteb import MTEB

benchmark = MTEB()
# Get retrieval tasks
retrieval_tasks = [t for t in benchmark.tasks if "Retrieval" in t.description]

# Evaluate and export to TREC format
for task in retrieval_tasks:
    # ... run evaluation and export
```

## Recommended Evaluation Sequence

### Week 1-2: Foundation
1. MS MARCO (establish baseline)
2. BEIR core (3-5 datasets)

### Week 3-4: Expansion
3. TREC Deep Learning
4. LoTTE
5. MIRACL (2-3 languages)

### Week 5-6: Domain Specialization
6. LegalBench-RAG
7. FiQA
8. BioASQ or SciFact-Open

### Week 7-8: Question Answering
9. HotpotQA
10. Natural Questions
11. SQuAD

### Week 9-10: Advanced
12. MTEB retrieval tasks
13. Regional datasets (if accessible)
14. Specialized datasets (IFIR, ANTIQUE, etc.)

## Research Questions by Dataset Category

### Multilingual Evaluation
- **MIRACL**: Which fusion methods work best across languages?
- **MTEB**: How does fusion perform across 112 languages?
- **FIRE/CLEF/NTCIR**: Regional language fusion effectiveness?

### Domain-Specific Evaluation
- **LegalBench-RAG**: Does fusion improve precise legal retrieval?
- **FiQA**: How does fusion handle financial opinion content?
- **BioASQ**: Biomedical domain fusion strategies?
- **SciFact-Open**: Scientific evidence retrieval fusion?

### Complex Retrieval
- **HotpotQA**: Multi-hop retrieval fusion?
- **IFIR**: Instruction-following fusion effectiveness?
- **ANTIQUE**: Non-factoid fusion strategies?

### Specialized Scenarios
- **FULTR**: Satisfaction-oriented fusion?
- **BordIRlines**: Bias-aware cross-lingual fusion?
- **TREC-COVID**: Crisis information fusion?

## Implementation Notes

### Multilingual Datasets
- MIRACL and MTEB require language-specific handling
- Consider language-specific fusion configurations
- Evaluate cross-lingual fusion effectiveness

### Domain-Specific Datasets
- May require domain-specific preprocessing
- Consider domain-specific fusion weights
- Evaluate domain expertise integration

### Large-Scale Datasets
- Natural Questions (42GB) requires efficient processing
- MTEB (58 datasets) needs systematic evaluation
- Consider sampling strategies for very large datasets

### Format Conversion
- Most HuggingFace datasets need TREC conversion
- Create conversion utilities for common formats
- Validate conversion accuracy

## Expected Insights by Dataset

### MIRACL
- Language-specific fusion preferences
- Cross-lingual fusion effectiveness
- Multilingual generalization patterns

### MTEB
- Task-specific fusion effectiveness
- Domain-specific fusion (legal, code, healthcare)
- Multilingual embedding fusion

### LegalBench-RAG
- Precise retrieval fusion
- Legal terminology handling
- Domain-specific fusion weights

### HotpotQA
- Multi-hop fusion strategies
- Multi-document fusion
- Explainable fusion approaches

## Next Steps

1. **Start with Priority 1-2 datasets** (MS MARCO, BEIR, TREC, LoTTE)
2. **Expand to multilingual** (MIRACL, MTEB)
3. **Add domain-specific** (Legal, Financial, Biomedical)
4. **Evaluate complex scenarios** (Multi-hop, Instruction-following)
5. **Regional evaluation** (FIRE, CLEF, NTCIR if accessible)

See `DATASET_RECOMMENDATIONS.md` for detailed information on each dataset.

