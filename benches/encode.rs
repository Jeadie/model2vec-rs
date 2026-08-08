// Standard encode-throughput benchmark for the real `StaticModel::encode`
// path (see optimisations.md for the hypotheses this validated: H1's custom
// WordPiece tokenizer and H4's contiguous-slice pooling, both now the only
// implementation — no feature flags to choose between).
//
//   cargo bench --bench encode -- 3
//   cargo bench --bench encode -- 3 minishlab/potion-retrieval-32M
//
// Reports single-query throughput and a batch-size sweep (best-of-reps).
// Reads big.txt from go-potion's shared cache (run the Go benchmark first
// to populate it), matching the other cross-implementation benchmarks.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use model2vec_rs::model::StaticModel;

fn main() {
    // `cargo bench` injects libtest flags (e.g. `--bench`); keep only positionals.
    let pos: Vec<String> = env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
    let reps: usize = pos.first().and_then(|s| s.parse().ok()).unwrap_or(3);
    let model_id = pos
        .get(1)
        .cloned()
        .unwrap_or_else(|| "minishlab/potion-base-2M".to_string());

    let cache = cache_dir();
    let text = fs::read_to_string(format!("{cache}/big.txt"))
        .expect("big.txt not found in cache; run the Go benchmark first");
    let words: Vec<&str> = text.split_whitespace().collect();
    let chunks: Vec<String> = words.chunks(256).map(|c| c.join(" ")).collect();
    let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();

    let model = StaticModel::from_pretrained(&model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));

    println!(
        "model={model_id}  chunks={}  bytes={}  reps={reps} (best-of)\n",
        chunks.len(),
        total_bytes
    );

    // Single-query: one chunk per encode call.
    let single = best_mbps(reps, total_bytes, || {
        let mut acc = 0.0_f32;
        for chunk in &chunks {
            acc += model.encode(std::slice::from_ref(chunk))[0][0];
        }
        black_box(acc);
    });
    println!("single-query   {single:>8.2} MB/s");

    // Batch-size sweep through the real batch encode.
    for &bs in &[1usize, 32, 256, 1024, chunks.len()] {
        let bs = bs.clamp(1, chunks.len());
        let name = if bs == chunks.len() {
            "all".to_string()
        } else {
            bs.to_string()
        };
        let mbps = best_mbps(reps, total_bytes, || {
            let mut acc = 0.0_f32;
            for group in chunks.chunks(bs) {
                acc += model.encode(group)[0][0];
            }
            black_box(acc);
        });
        println!("batch={name:<9} {mbps:>8.2} MB/s");
    }
}

// best_mbps runs `work` `reps` times and returns the throughput of the fastest
// run (least noise), in MB/s over `total_bytes`.
fn best_mbps(reps: usize, total_bytes: usize, mut work: impl FnMut()) -> f64 {
    work(); // warm up
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        work();
        best = best.min(start.elapsed().as_secs_f64());
    }
    total_bytes as f64 / best / 1024.0 / 1024.0
}

// cache_dir mirrors go-potion's resolveCacheDir: GO_POTION_HOME, then the
// platform user cache directory.
fn cache_dir() -> String {
    if let Ok(dir) = env::var("GO_POTION_HOME") {
        if !dir.is_empty() {
            return dir;
        }
    }
    if cfg!(target_os = "macos") {
        let home = env::var("HOME").expect("HOME not set");
        return format!("{home}/Library/Caches/go-potion");
    }
    if cfg!(target_os = "windows") {
        let local = env::var("LOCALAPPDATA").expect("LOCALAPPDATA not set");
        return format!("{local}\\go-potion");
    }
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return format!("{xdg}/go-potion");
        }
    }
    let home = env::var("HOME").expect("HOME not set");
    format!("{home}/.cache/go-potion")
}
