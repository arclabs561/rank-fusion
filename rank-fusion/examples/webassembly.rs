//! Example: WebAssembly usage of rank-fusion (conceptual).
//!
//! This demonstrates how rank-fusion could be used in browser-based RAG applications.
//!
//! # Building for WebAssembly
//!
//! ```bash
//! # Install wasm-pack
//! curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
//!
//! # Build for web
//! wasm-pack build --target web --out-dir pkg
//! ```
//!
//! # JavaScript Usage (conceptual)
//!
//! ```javascript
//! import init, { rrf } from './pkg/rank_fusion.js';
//!
//! async function run() {
//!     await init();
//!
//!     const bm25 = [["d1", 12.5], ["d2", 11.0]];
//!     const dense = [["d2", 0.9], ["d3", 0.8]];
//!
//!     const fused = rrf(bm25, dense, 60);
//!     // [["d2", 0.033], ["d1", 0.016], ["d3", 0.016]]
//!
//!     console.log("Fused results:", fused);
//! }
//!
//! run();
//! ```
//!
//! # Use Cases
//!
//! - Browser-based RAG applications
//! - Client-side search result fusion
//! - Offline-capable search systems
//! - Privacy-preserving search (no server round-trip)

fn main() {
    println!("This is a conceptual example of WebAssembly integration.");
    println!("See the source code comments for JavaScript API design.");
    println!("\nTo build for WebAssembly:");
    println!("1. Install wasm-pack: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh");
    println!("2. Build: wasm-pack build --target web --out-dir pkg");
    println!("3. Use from JavaScript: import {{ rrf }} from './pkg/rank_fusion.js'");
}
