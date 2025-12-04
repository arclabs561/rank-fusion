# User Personas for rank-fusion and rank-refine

## Repository Purposes

### rank-fusion
**Purpose**: Combine ranked lists from multiple retrievers (BM25, dense embeddings, sparse vectors) into a single ranking.

**Core Problem Solved**: Different retrievers use incompatible score scales (e.g., BM25: 0-100, dense: 0-1). Normalization is fragile and requires tuning.

**Key Value**: Zero-configuration fusion that finds consensus across retrievers without normalization.

### rank-refine
**Purpose**: SIMD-accelerated similarity scoring for vector search and RAG with token-level precision.

**Core Problem Solved**: Dense retrieval loses token-level alignment. A query like "capital of France" might match "France's economic capital" incorrectly.

**Key Value**: Late interaction (ColBERT-style) preserves token-level semantics for precision-critical reranking.

---

## Prioritized User Personas

### Tier 1: Primary Users (Highest Priority)

#### 1. **RAG Pipeline Engineers at AI Startups** ⭐⭐⭐
**Profile**: Early-stage AI companies building production RAG systems (similar to Assembled's deployment)
- **Background**: 2-5 years ML engineering, Python-first, learning Rust for performance
- **Company Context**: 10-50 person startup, need to ship fast, limited infrastructure budget
- **Tech Stack**: Python (primary), Rust (hot paths), PostgreSQL + pgvector, OpenAI/Anthropic APIs
- **Use Cases**:
  - `rank-fusion`: Combining BM25 + dense retrieval results (like Assembled's hybrid search)
  - `rank-refine`: Reranking 100-1000 candidates with ColBERT (61ms per query target)
  - Explainability: Debugging why certain docs ranked low (critical for customer demos)
- **Pain Points**: 
  - Retrieval quality directly impacts LLM output (customer complaints about hallucinations)
  - Need to understand why documents were selected (support tickets)
  - Latency constraints (real-time chat, <200ms total)
  - Cost constraints (can't afford expensive cross-encoders for all queries)
- **Decision Drivers**: 
  - Zero-configuration fusion (RRF) - no time to tune normalization
  - Python bindings essential (team doesn't know Rust yet)
  - Performance benchmarks matter (need <100ms reranking)
  - Explainability for debugging customer issues
- **Real-World Context**: Assembled deployed RRF in production after testing, found it outperformed complex methods
- **Likelihood**: **Very High** - RAG is #1 use case, startups are early adopters

#### 2. **Search Infrastructure Engineers at Enterprise Platforms** ⭐⭐⭐
**Profile**: Engineers building search infrastructure (like OpenSearch, Azure AI Search teams)
- **Background**: 5-10 years systems engineering, Rust/C++ expertise, performance-critical mindset
- **Company Context**: Large enterprise (Microsoft, AWS, Elastic), building platforms for customers
- **Tech Stack**: Rust (primary), Java/Scala (integration), C++ (legacy)
- **Use Cases**:
  - `rank-fusion`: Core fusion engine for hybrid search (OpenSearch neural search plugin style)
  - `rank-refine`: Optional reranking stage (customers choose)
  - Integration with existing search stacks (Elasticsearch, OpenSearch)
- **Pain Points**:
  - Score scale incompatibility (BM25 vs embeddings) - customer problem
  - Need zero-configuration fusion (RRF) - can't require tuning
  - Performance at scale (100-1000 item lists, millions of queries/day)
  - Vendoring requirements (can't add dependencies to core platform)
- **Decision Drivers**:
  - Zero dependencies (vendoring-friendly) - critical for platform code
  - Performance (13μs for 100 items) - must handle high throughput
  - Self-contained code (~2000 lines) - easy to review/maintain
  - Multiple algorithm options (RRF, CombSUM, etc.) - customer choice
- **Real-World Context**: OpenSearch incorporated RRF into neural search plugin (RFC August 2024), Azure AI Search uses RRF in production
- **Likelihood**: **Very High** - This is the core purpose, enterprise platforms are adopting

#### 3. **E-commerce Search Product Engineers** ⭐⭐⭐
**Profile**: Product engineers at e-commerce companies (Shopee, Amazon-style teams)
- **Background**: 3-7 years full-stack + ML, Python/Java, business metrics focused
- **Company Context**: Mid-size to large e-commerce (100-1000 employees), revenue-driven
- **Tech Stack**: Python (ML), Java/Scala (serving), Redis, Elasticsearch
- **Use Cases**:
  - `rank-fusion`: `additive_multi_task` for CTR + CTCVR fusion (ResFlow: 1.29% OPU increase)
  - `rank-refine`: Diversity selection (MMR) to avoid duplicate products
  - A/B testing different fusion strategies (product manager requests)
- **Pain Points**:
  - Multi-task ranking (CTR, conversion, revenue) - business metrics
  - Need to avoid showing 10 identical products (user complaints)
  - Long-tail queries lack behavioral data (BEQUE-style query rewriting needed)
  - A/B testing infrastructure (need to compare algorithms)
- **Decision Drivers**:
  - Validated algorithms (ResFlow shows 1.29% OPU increase) - business case
  - Real-time performance (affects conversion rates)
  - Explainability for product managers (need to justify changes)
  - Python bindings (ML team uses Python)
- **Real-World Context**: Shopee production A/B tests validated ResFlow approach, Siksilk achieved 25% conversion increase with optimized search
- **Likelihood**: **High** - Specific e-commerce features, validated business impact

---

### Tier 2: Secondary Users (High Priority)

#### 4. **ML Engineers at Research Labs** ⭐⭐
**Profile**: ML engineers implementing IR research papers (SIGIR, ICML, NeurIPS)
- **Background**: PhD or MS in CS/ML, 2-4 years research experience, Python expert
- **Company Context**: Research labs (academic or industry), publication-focused
- **Tech Stack**: Python (primary), PyTorch/TensorFlow, Jupyter notebooks
- **Use Cases**:
  - `rank-fusion`: Hyperparameter optimization, algorithm comparison (BEIR benchmarks)
  - `rank-refine`: Token pooling experiments, Matryoshka refinement
  - Reproducing paper results, implementing baselines
- **Pain Points**:
  - Need reproducible benchmarks (BEIR, MS MARCO)
  - Want to understand algorithm trade-offs (RRF vs CombSUM)
  - Need to implement custom fusion strategies (paper experiments)
  - Out-of-domain generalization testing (BEIR shows in-domain ≠ out-of-domain)
- **Decision Drivers**:
  - Well-documented formulas and research citations
  - IR metrics (NDCG, MRR, recall) - standard evaluation
  - Hyperparameter optimization tools
  - Python-first (research codebase)
- **Real-World Context**: BEIR benchmark revealed RRF generalization issues, researchers need diverse evaluation
- **Likelihood**: **High** - Research papers cited, evaluation metrics included, active research community

#### 5. **Legal Tech Software Engineers** ⭐⭐
**Profile**: Engineers building legal document search (case law, contract analysis)
- **Background**: 3-6 years software engineering, some domain knowledge, security-conscious
- **Company Context**: Legal tech startups (50-200 employees), compliance-focused
- **Tech Stack**: Python (primary), TypeScript (frontend), PostgreSQL, Elasticsearch
- **Use Cases**:
  - `rank-refine`: ColBERT reranking for token-level precision ("capital" vs "economic capital")
  - `rank-fusion`: Combining multiple legal document retrievers (case law + statutes)
  - Explainability: Show why documents were selected (legal discovery)
- **Pain Points**:
  - Cannot afford false positives (wrong case citations)
  - Need to explain why documents were selected (legal discovery requirements)
  - Token-level alignment critical (precise phrase matching)
  - HIPAA compliance (if medical records involved)
- **Decision Drivers**:
  - Precision over recall (false positives are costly)
  - Explainability features (legal discovery)
  - Token-level alignment/highlighting (show exact matches)
  - Python bindings (team uses Python)
- **Real-World Context**: Legal professionals need accurate patient information, comprehensive provider identification, record authenticity verification
- **Likelihood**: **Medium-High** - Docs mention "precision-critical applications", legal tech is growing

#### 6. **Data Scientists Building Search Features** ⭐⭐
**Profile**: Data scientists at companies adding search to existing products
- **Background**: 2-5 years data science, Python/scikit-learn expert, less systems experience
- **Company Context**: Mid-size companies (200-2000 employees), adding search to existing product
- **Tech Stack**: Python (exclusive), scikit-learn, pandas, Jupyter
- **Use Cases**:
  - `rank-fusion-python`: Python bindings for fusion (don't know Rust)
  - `rank-refine-python`: Python bindings for reranking
  - Feature engineering for ranking models
- **Pain Points**:
  - Don't want to learn Rust (time constraint)
  - Need easy Python API (scikit-learn style)
  - Performance less critical than ease of use (internal tools)
  - Integration with existing ML pipelines
- **Decision Drivers**:
  - Python package availability (PyPI) - essential
  - Simple API (familiar patterns)
  - Good documentation (self-service)
  - Performance acceptable (not critical path)
- **Real-World Context**: 10.6% of developers use scikit-learn (Stack Overflow 2024), data scientists prefer Python-first tools
- **Likelihood**: **Medium-High** - Python bindings exist, large Python ML community

---

### Tier 3: Tertiary Users (Medium Priority)

#### 7. **Full-Stack Developers Building Search UIs** ⭐
**Profile**: Frontend/full-stack developers adding search to web apps
- **Background**: 2-5 years web development, JavaScript/TypeScript expert, performance-aware
- **Company Context**: SaaS companies, adding search to existing products
- **Tech Stack**: TypeScript/JavaScript, React/Vue, WebAssembly (optional)
- **Use Cases**:
  - `rank-fusion` WASM: Client-side fusion for hybrid search (if needed)
  - `rank-refine` WASM: Client-side reranking (rare, usually server-side)
- **Pain Points**:
  - Need browser-compatible code (WASM)
  - Bundle size constraints (WASM not always faster than JS)
  - Real-time user interactions (typing, autocomplete)
  - DOM binding overhead (WASM performance gains limited)
- **Decision Drivers**:
  - WASM package availability (npm) - if needed
  - Small bundle size (optimized WASM) - important
  - TypeScript definitions - essential
  - Performance reality: WASM 1.75-2.5x slower than native, variable across browsers
- **Real-World Context**: WASM performance is nuanced - write-once portability more valuable than speed, DOM overhead limits gains
- **Likelihood**: **Low-Medium** - WASM support exists but server-side more common, client-side search usually simpler

#### 8. **Recommendation System Engineers at Media Companies** ⭐
**Profile**: Engineers building recommendation systems (Netflix, Spotify-style)
- **Background**: 3-7 years ML/systems engineering, Python + Scala/Java
- **Company Context**: Media/entertainment companies, engagement-driven
- **Tech Stack**: Python (ML), Scala/Java (serving), Spark, Redis
- **Use Cases**:
  - `rank-refine`: Diversity selection (MMR, DPP) to avoid redundancy
  - `rank-fusion`: Combining multiple recommendation signals (collaborative + content-based)
- **Pain Points**:
  - Need diverse results (not 10 identical items) - user engagement
  - Multiple recommendation strategies (A/B testing)
  - Heat-spreading algorithms for novelty
  - Personalized adaptive diversity (different users want different diversity)
- **Decision Drivers**:
  - MMR/DPP algorithms - proven approaches
  - Performance (real-time serving)
  - Python bindings (ML team)
- **Real-World Context**: Diversity-accuracy dilemma resolved through hybrid algorithms, users engage as knowledge-exploration task
- **Likelihood**: **Medium-Low** - Mentioned in docs but not primary focus, recommendation systems have specialized needs

#### 9. **Multimodal Search Developers (ColPali)** ⭐
**Profile**: Engineers building vision-language search (document images, medical imaging)
- **Background**: 4-8 years ML engineering, computer vision + NLP, research-oriented
- **Company Context**: Specialized companies (document AI, medical imaging), niche market
- **Tech Stack**: Python (primary), PyTorch, Rust (hot paths)
- **Use Cases**:
  - `rank-refine`: Text-to-image alignment, visual snippet extraction
  - Image patch embeddings as "tokens"
  - Visual snippet extraction for document images
- **Pain Points**:
  - Token-level alignment for image patches (32×32 grid = 1024 patches/page)
  - Visual snippet extraction (show relevant image regions)
  - Multimodal RAG (VisRAG shows 20-40% gains over text-only)
- **Decision Drivers**:
  - ColPali support (text-to-image alignment)
  - `patches_to_regions()` utility (convert patches to pixel coordinates)
  - Token-level alignment functions (work for both text and images)
- **Real-World Context**: VisRAG achieves 20-40% end-to-end performance gains, multimodal RAG is emerging
- **Likelihood**: **Low-Medium** - Specific feature, niche use case, but growing (multimodal RAG emerging)

#### 10. **DevOps/MLOps Engineers** ⭐
**Profile**: Engineers deploying and monitoring search systems
- **Background**: 3-6 years DevOps/MLOps, infrastructure focus, less algorithm knowledge
- **Company Context**: Any company with production ML systems
- **Tech Stack**: Kubernetes, Docker, monitoring tools, Python (scripts)
- **Use Cases**:
  - Deploying `rank-fusion` and `rank-refine` in production
  - Performance monitoring, A/B testing infrastructure
  - Cost optimization (ColBERT 2 orders of magnitude faster than cross-encoders)
- **Pain Points**:
  - Deployment complexity (Rust binaries, Python bindings)
  - Performance monitoring (latency, throughput)
  - Cost optimization (reranking is expensive)
  - Model versioning and retraining
- **Decision Drivers**:
  - Easy deployment (Docker images, package managers)
  - Performance characteristics (need to plan capacity)
  - Resource requirements (memory, CPU)
  - Observability (metrics, logging)
- **Real-World Context**: ColBERT is 2 orders of magnitude faster than cross-encoders (61ms vs much slower), infrastructure matters
- **Likelihood**: **Low** - Indirect users, but critical for production adoption

---

### Tier 4: Edge Cases (Lower Priority)

#### 11. **Startup Founders Building MVP Search**
**Profile**: Technical founders building MVP with search features
- **Background**: Technical but time-constrained, need to ship fast
- **Company Context**: Early-stage startup (1-10 employees), MVP phase
- **Tech Stack**: Whatever works fastest (Python, JavaScript, etc.)
- **Use Cases**:
  - `rank-fusion`: Quick hybrid search implementation (zero-config RRF)
  - `rank-refine`: Optional, only if needed for MVP
- **Pain Points**:
  - Need to validate search concept quickly
  - Limited resources (can't build custom solutions)
  - Must-have features only (MoSCoW prioritization)
- **Decision Drivers**:
  - Zero-configuration (no time to tune)
  - Fast implementation (Python bindings)
  - Good enough performance (not optimizing yet)
- **Real-World Context**: MVP development focuses on problem validation, rapid iteration, early adopter feedback
- **Likelihood**: **Low-Medium** - Startups are early adopters but may use simpler solutions initially

#### 12. **Graduate Students / Researchers**
**Profile**: Students learning IR algorithms, implementing papers
- **Background**: MS/PhD students, learning IR, Python-focused
- **Company Context**: Universities, research labs
- **Tech Stack**: Python, Jupyter notebooks, PyTorch
- **Use Cases**:
  - Understanding fusion algorithms (RRF, CombSUM, etc.)
  - Educational examples, reproducing results
  - Implementing custom variants for research
- **Pain Points**:
  - Need clear documentation and examples
  - Want to understand algorithms deeply
  - May need to modify/extend code
- **Decision Drivers**:
  - Well-documented formulas
  - Educational examples
  - Python bindings (learning tool)
- **Real-World Context**: Students are future practitioners, good documentation helps adoption
- **Likelihood**: **Low** - Well-documented but not primary audience, important for long-term adoption

---

## Usage Pattern Analysis

### Most Common Workflow (Production)
1. **Retrieve** → Multiple retrievers (BM25, dense, sparse) - parallel execution
2. **Fuse** → `rank-fusion` combines results (RRF k=60 default, or CombSUM if scales match)
3. **Refine** → `rank-refine` reranks top 100-1000 candidates (ColBERT, 61ms target)
4. **Diversify** → `rank-refine` MMR for diversity (optional, λ=0.5 default for RAG)
5. **Explain** → Debug/analyze results (optional, for production debugging)

### Real-World Adoption Patterns

**Early Adopters (2024)**:
- **Assembled**: Deployed RRF in production, found it outperformed complex methods
- **OpenSearch**: Incorporated RRF into neural search plugin (RFC August 2024)
- **Azure AI Search**: Uses RRF in production hybrid search
- **Infinity/Vespa**: Database systems with tensor support for late interaction

**Adoption Timeline**:
- **2024**: Late interaction (ColBERT) transitioned from research to production
- **Mid-2024**: Database systems added native tensor support
- **Summer 2024**: Infrastructure components accelerated
- **Current**: Concentrated among organizations with advanced retrieval infrastructure

### Language Distribution (Real Usage)
- **Rust**: Infrastructure/platform engineers (OpenSearch, Azure teams) - performance-critical
- **Python**: ML engineers, data scientists, researchers - 10.6% use scikit-learn (Stack Overflow 2024)
- **JavaScript/TypeScript**: Frontend developers - WASM for portability, not always performance
- **Java/Scala**: Enterprise integration (Elasticsearch, OpenSearch ecosystems)

### Organization Size & Maturity
- **Startups (10-50)**: High adoption (Assembled-style), need zero-config, fast iteration
- **Mid-size (50-200)**: Medium adoption, building production systems, Python-first
- **Large enterprises (200+)**: Medium adoption, infrastructure teams, Rust/C++ expertise
- **Research labs**: High adoption, well-documented algorithms, Python-focused

### Performance Constraints (Real-World)
- **RAG systems**: <200ms total latency (real-time chat)
- **ColBERT reranking**: 61ms per query target (2 orders of magnitude faster than cross-encoders)
- **Fusion**: 13μs for 100 items (suitable for real-time)
- **WASM**: 1.75-2.5x slower than native, variable across browsers (write-once portability value)

---

## Prioritization Summary

**Top 3 Personas (Focus 80% effort here):**
1. **RAG Pipeline Engineers at AI Startups** - Early adopters, Python-first, need zero-config
2. **Search Infrastructure Engineers at Enterprise Platforms** - Core users, Rust-first, need vendoring
3. **E-commerce Search Product Engineers** - Business-driven, validated algorithms, Python ML teams

**Secondary Personas (Focus 15% effort):**
4. **ML Engineers at Research Labs** - Reproducibility, benchmarks, Python-focused
5. **Legal Tech Software Engineers** - Precision-critical, explainability, Python-first
6. **Data Scientists Building Search Features** - Python-exclusive, ease of use, scikit-learn style

**Tertiary Personas (Focus 5% effort):**
7-10. Frontend developers, recommendation engineers, multimodal developers, MLOps
11-12. Startup founders, students (long-term adoption)

---

## Real-World Adoption Insights

### Production Deployments (2024)
- **Assembled**: RRF in production, outperformed complex methods
- **OpenSearch**: RRF in neural search plugin (RFC August 2024)
- **Azure AI Search**: RRF in production hybrid search
- **Infinity/Vespa**: Database systems with tensor support for late interaction

### Why Production Teams Choose These Tools
1. **Zero-configuration** (RRF) - No tuning needed, works across diverse data
2. **Performance** - ColBERT 2 orders of magnitude faster than cross-encoders
3. **Robustness** - RRF handles unrelated scores, minimal parameter tuning
4. **Vendoring-friendly** - Zero dependencies, self-contained code
5. **Validated algorithms** - ResFlow (1.29% OPU), ERANK (2-5% NDCG improvement)

### Adoption Barriers
1. **Rust learning curve** - Python-first teams need bindings
2. **Late interaction complexity** - ColBERT requires token embeddings (10-50x storage)
3. **WASM performance reality** - Not always faster, portability more valuable
4. **Out-of-domain generalization** - BEIR shows in-domain ≠ out-of-domain performance

---

## Recommendations for Documentation/Marketing

### Primary Focus (Tier 1)
1. **Lead with RAG use case** - Most common, highest value (Assembled example)
2. **Emphasize zero-configuration** - Key differentiator (RRF k=60 default)
3. **Show complete pipeline** - Retrieve → Fuse → Refine → Explain (real workflow)
4. **Python-first examples** - Lower barrier to entry (10.6% use scikit-learn)
5. **Performance benchmarks** - Critical for infrastructure teams (13μs, 61ms targets)
6. **Real production examples** - Assembled, OpenSearch, Azure AI Search deployments

### Secondary Focus (Tier 2)
7. **Research citations** - BEIR, MS MARCO, paper references (researchers need this)
8. **Explainability features** - Debugging production issues (customer support)
9. **Vendoring documentation** - Self-contained code, zero dependencies (platform teams)
10. **Out-of-domain evaluation** - BEIR benchmarks, generalization testing

### Tertiary Focus (Tier 3)
11. **WASM realistic expectations** - Portability over performance, browser variability
12. **Diversity algorithms** - MMR/DPP for recommendation systems
13. **Multimodal examples** - ColPali, VisRAG (emerging use case)

