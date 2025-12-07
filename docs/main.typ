#set page(margin: (x: 2.5cm, y: 2cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.65em)

#show heading: set text(weight: "bold")

= rank-fusion: Rank Fusion Algorithms Documentation

#align(center)[
  #text(size: 14pt, weight: "bold")[Rank Fusion Algorithms for Information Retrieval]
  
  #text(size: 10pt)[Combining Multiple Ranking Systems]
  
  #v(0.5cm)
  #text(size: 9pt, style: "italic")[Version 0.1.0]
]

== Introduction

`rank-fusion` provides efficient implementations of rank fusion algorithms for combining multiple ranking systems. The primary algorithm is Reciprocal Rank Fusion (RRF), which is particularly effective for combining rankings with incompatible score scales.

== Features

- *Reciprocal Rank Fusion (RRF)*: Primary fusion method, works with incompatible score scales
- *Multiple Algorithms*: ISR, CombSUM, CombMNZ, Borda, DBSF, Weighted, Standardized, Additive Multi-Task
- *Performance Optimized*: < 1ms for 1000 items
- *Evaluation Integration*: Works seamlessly with rank-eval

== Quick Start

```rust
use rank_fusion::rrf;

let rankings = vec![
    vec!["doc1", "doc2", "doc3", "doc4"],
    vec!["doc3", "doc1", "doc4", "doc2"],
    vec!["doc2", "doc3", "doc1", "doc4"],
];

let fused = rrf(&rankings, 60); // k=60 (default)
// Result: ["doc1", "doc3", "doc2", "doc4"] (approximate)
```

== Reciprocal Rank Fusion (RRF)

RRF is the recommended fusion method when combining rankings with incompatible score scales. It uses reciprocal rank scoring:

$ "RRF"(d) = sum_(r "in" R) 1 / (k + "rank"_r(d)) $

#v(0.5em)
Parameters: R is the set of rankings to fuse; k is a constant, default value 60; rank of d for r is the position of document d for ranking r.

=== Why RRF?

#v(0.3em)
- Score Scale Independent: Works with any scoring system
- Robust: Handles missing documents gracefully
- Effective: Proven to outperform individual rankers
- Fast: O(n × m) where n is documents, m is rankings

=== Parameter Tuning

#v(0.3em)
- k from 20 to 40: Strong consensus, few top documents dominate
- k equals 60: Default, balanced
- k equals 100 and above: Broad agreement, more documents contribute

== Other Fusion Methods

=== CombSUM

Simple sum of scores (requires compatible scales):

$ "CombSUM"(d) = sum_(r "in" R) "score"_r(d) $

=== CombMNZ

Sum multiplied by number of rankings containing document:

$ "CombMNZ"(d) = |R_d| * sum_(r "in" R) "score"_r(d) $

where R_d = rankings containing document d.

=== Borda Count

Sum of inverse ranks:

$ "Borda"(d) = sum_(r "in" R) (n - "rank"_r(d) + 1) $

where n = number of documents.

== Performance

- RRF: < 1ms for 1000 items, 10 rankings
- CombSUM: < 0.5ms for 1000 items
- All methods have linear scaling with number of documents

== Evaluation

Integration with rank-eval:

```rust
use rank_fusion::rrf;
use rank_eval::ndcg_at_k;

let fused = rrf(&rankings, 60);
let ndcg = ndcg_at_k(&fused, &relevant_docs, 10);
```

== Statistical Analysis

Real evaluation data shows:

- RRF consistently outperforms individual rankers
- Optimal k parameter varies by dataset
- Method comparison visualizations are available within hack/viz directory

== Installation

```bash
cargo add rank-fusion
```

== Examples

See examples/ directory for complete workflows.

== References

=== Primary Sources

- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods". In *Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval* (pp. 758-759).

=== Evaluation Standards

- TREC evaluation guidelines and standards for information retrieval evaluation
- Voorhees, E. M. (2004). "Overview of the TREC 2004 Robust Retrieval Track". In *TREC*.

== License

MIT OR Apache-2.0

