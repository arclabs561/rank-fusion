# Dataset Expansion Summary

## What Was Added

Expanded dataset recommendations from **6 datasets** to **21+ datasets** across 7 priority levels.

### New Datasets Added

#### Multilingual (Priority 3)
- **MIRACL** - 18 languages, 40k+ queries, human-annotated
- **MTEB** - 58 datasets, 112 languages, 8 task categories

#### Domain-Specific (Priority 4)
- **LegalBench-RAG** - Legal domain, precise retrieval
- **FiQA** - Financial domain, opinion-based QA
- **BioASQ** - Biomedical domain, structured + unstructured
- **SciFact-Open** - Scientific claim verification, 500k abstracts

#### Question Answering (Priority 5)
- **HotpotQA** - Multi-hop reasoning, 112k+ queries
- **Natural Questions** - Real Google queries, 42GB
- **SQuAD** - Reading comprehension, 107k+ queries

#### Regional/Language-Specific (Priority 6)
- **FIRE** - South Asian languages
- **CLEF** - European languages, multimodal
- **NTCIR** - Asian languages, cross-lingual

#### Specialized (Priority 7)
- **IFIR** - Instruction-following IR, 4 domains
- **ANTIQUE** - Non-factoid questions, 2.6k queries
- **BordIRlines** - Cross-lingual geopolitical bias

## Updated Files

1. **`DATASET_RECOMMENDATIONS.md`** - Expanded with all 21+ datasets
2. **`EXTENDED_DATASET_GUIDE.md`** - New comprehensive guide
3. **`dataset_loaders.rs`** - Added MIRACL and MTEB loaders
4. **`scripts/download_miracl.sh`** - MIRACL download helper
5. **`scripts/download_mteb.sh`** - MTEB download helper

## Dataset Categories

### By Language Coverage
- **English-only**: MS MARCO, BEIR, TREC, LoTTE, LegalBench, FiQA, BioASQ, HotpotQA, NQ, SQuAD
- **Multilingual**: MIRACL (18), MTEB (112), FIRE, CLEF, NTCIR, BordIRlines

### By Domain
- **General**: MS MARCO, BEIR, TREC, LoTTE, MIRACL, MTEB
- **Legal**: LegalBench-RAG
- **Financial**: FiQA
- **Biomedical**: BioASQ, SciFact-Open, TREC-COVID
- **Scientific**: SciFact-Open
- **Question Answering**: HotpotQA, Natural Questions, SQuAD, ANTIQUE

### By Task Type
- **Ad-hoc Retrieval**: MS MARCO, BEIR, TREC, MIRACL
- **Question Answering**: HotpotQA, NQ, SQuAD, ANTIQUE
- **Multi-hop**: HotpotQA
- **Claim Verification**: SciFact-Open, FEVER
- **Instruction-following**: IFIR
- **Cross-lingual**: MIRACL, BordIRlines, FIRE, CLEF, NTCIR

## Access Methods

### HuggingFace (Easiest)
- MIRACL, LoTTE, HotpotQA, SQuAD, FiQA

### Python Frameworks
- MTEB: `pip install mteb`
- BEIR: `pip install beir`
- ir_datasets: `pip install ir_datasets`

### Direct Download
- MS MARCO, TREC, LegalBench, BioASQ, Natural Questions

### Regional Forums
- FIRE, CLEF, NTCIR

## Implementation Roadmap Updated

Now includes 5 phases covering all 21+ datasets:
- Phase 1: Foundation (MS MARCO, BEIR)
- Phase 2: Expansion (TREC, LoTTE, MIRACL)
- Phase 3: Domain Specialization (Legal, Financial, Biomedical)
- Phase 4: Advanced (QA, MTEB, Regional)
- Phase 5: Specialized (FULTR, IFIR, ANTIQUE, etc.)

## Research Coverage

The expanded dataset list enables evaluation of:
- ✅ Multilingual fusion (18-112 languages)
- ✅ Domain-specific fusion (Legal, Financial, Biomedical, Scientific)
- ✅ Complex retrieval (Multi-hop, Instruction-following, Non-factoid)
- ✅ Cross-lingual fusion (MIRACL, BordIRlines, Regional datasets)
- ✅ Regional language fusion (FIRE, CLEF, NTCIR)
- ✅ Specialized scenarios (Satisfaction-oriented, Bias-aware, Crisis retrieval)

## Next Steps

1. Start with Priority 1-2 datasets (MS MARCO, BEIR, TREC, LoTTE)
2. Expand to multilingual (MIRACL, MTEB)
3. Add domain-specific (Legal, Financial, Biomedical)
4. Evaluate complex scenarios (Multi-hop, Instruction-following)
5. Regional evaluation (FIRE, CLEF, NTCIR if accessible)

All datasets are documented with:
- Why they're valuable
- What they test
- Implementation notes
- Expected insights
- Dataset characteristics
- Access methods

See `DATASET_RECOMMENDATIONS.md` for complete details on all 21+ datasets.

