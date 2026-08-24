mod common;
use common::load_test_model;
use model2vec_rs::model::StaticModel;
use std::fs;

/// Test that encoding an empty input slice yields an empty output
#[test]
fn test_encode_empty_input() {
    let model = load_test_model();
    let embs: Vec<Vec<f32>> = model.encode(&[]);
    assert!(embs.is_empty(), "Expected no embeddings for empty input");
}

/// Test that encoding a single empty sentence produces a zero vector
#[test]
fn test_encode_empty_sentence() {
    let model = load_test_model();
    let embs = model.encode(&["".to_string()]);
    assert_eq!(embs.len(), 1);
    let vec = &embs[0];
    assert!(vec.iter().all(|&x| x == 0.0), "All entries should be zero");
}

/// Test that encoding a single sentence returns the correct shape
#[test]
fn test_encode_single() {
    let model = load_test_model();
    let sentence = "hello world";

    // Single-sentence helper → 1-D
    let one_d = model.encode_single(sentence);

    // Batch call with a 1-element slice → 2-D wrapper
    let two_d = model.encode(&[sentence.to_string()]);

    // Shape assertions
    assert!(!one_d.is_empty(), "encode_single must return a non-empty 1-D vector");
    assert_eq!(
        two_d.len(),
        1,
        "encode(&[..]) should wrap the result in a Vec with length 1"
    );
    assert_eq!(
        two_d[0].len(),
        one_d.len(),
        "inner vector dimensionality should match encode_single output"
    );
}

/// Test override of `normalize` flag in from_pretrained
#[test]
fn test_normalization_flag_override() {
    // Load with normalize = true (default in config)
    let model_norm = StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, None, None).unwrap();
    let emb_norm = model_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_norm = emb_norm.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Load with normalize = false override
    let model_no_norm =
        StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, Some(false), None).unwrap();
    let emb_no = model_no_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_no = emb_no.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Normalized version should have unit length, override should give larger norm
    assert!(
        (norm_norm - 1.0).abs() < 1e-5,
        "Normalized vector should have unit norm"
    );
    assert!(
        norm_no > norm_norm,
        "Without normalization override, norm should be larger"
    );
}

/// Test from_borrowed constructor (zero-copy path)
#[test]
fn test_from_borrowed() {
    use safetensors::SafeTensors;
    use tokenizers::Tokenizer;

    let path = "tests/fixtures/test-model-float32";
    let tokenizer = Tokenizer::from_file(format!("{path}/tokenizer.json")).unwrap();
    let bytes = fs::read(format!("{path}/model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor("embeddings").unwrap();
    let [rows, cols]: [usize; 2] = tensor.shape().try_into().unwrap();
    let floats: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Leak to get 'static lifetime (fine for tests)
    let floats: &'static [f32] = Box::leak(floats.into_boxed_slice());

    let model = StaticModel::from_borrowed(tokenizer, floats, rows, cols, true, None, None).unwrap();
    let emb = model.encode_single("hello");
    assert!(!emb.is_empty());
}

/// Some sentence-transformers exports don't ship a config.json at all.
/// `from_pretrained` must still load the model, defaulting normalize=true,
/// instead of failing because config.json is missing.
#[test]
fn test_from_pretrained_without_config_json() {
    let source = "tests/fixtures/test-model-float32";
    let dir = std::env::temp_dir().join(format!("model2vec-rs-test-no-config-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::copy(format!("{source}/tokenizer.json"), dir.join("tokenizer.json")).unwrap();
    fs::copy(format!("{source}/model.safetensors"), dir.join("model.safetensors")).unwrap();
    // Deliberately no config.json in `dir`.

    let model = StaticModel::from_pretrained(&dir, None, None, None)
        .expect("loading a model with no config.json should succeed, defaulting normalize=true");
    let emb = model.encode_single("hello world");
    assert!(!emb.is_empty());

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_from_bytes_matches_from_pretrained_for_local_model() {
    let path = "tests/fixtures/test-model-float32";
    let from_path = StaticModel::from_pretrained(path, None, None, None).unwrap();
    let from_bytes = StaticModel::from_bytes(
        fs::read(format!("{path}/tokenizer.json")).unwrap(),
        fs::read(format!("{path}/model.safetensors")).unwrap(),
        Some(fs::read(format!("{path}/config.json")).unwrap()),
        None,
    )
    .unwrap();

    let query = "hello world";
    let path_embedding = from_path.encode_single(query);
    let bytes_embedding = from_bytes.encode_single(query);

    assert_eq!(path_embedding.len(), bytes_embedding.len());
    for (left, right) in path_embedding.iter().zip(bytes_embedding.iter()) {
        assert!(
            (left - right).abs() < 1e-6,
            "expected byte-loaded model to match path-loaded model"
        );
    }
}

/// A tokenizer whose `model.unk_token` is genuinely absent (`null`) must load
/// without error, and `StaticModel` must record `unk_token_id: None` rather
/// than defaulting to a literal "[UNK]" token and failing when that string
/// isn't in the vocab. This matches byte-level BPE tokenizers (e.g. Qwen's),
/// which have full byte coverage and never declare an unk token.
#[test]
fn test_load_tokenizer_without_unk_token() {
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file("tests/fixtures/byte-bpe-no-unk-tokenizer.json")
        .expect("tokenizer with unk_token: null should load");

    let rows = tokenizer.get_vocab_size(false);
    let cols = 4;
    let embeddings = vec![0.1_f32; rows * cols];

    let model = StaticModel::from_owned(tokenizer, embeddings, rows, cols, true, None, None)
        .expect("StaticModel should build even when the tokenizer has no unk_token");

    // Encoding should not panic or filter anything out due to a bogus unk id.
    let emb = model.encode_single("hello world");
    assert_eq!(emb.len(), cols);
}

#[cfg(all(not(feature = "hf-hub"), not(feature = "local-only")))]
#[test]
fn test_from_pretrained_remote_requires_hf_hub_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("hf-hub"),
        "expected remote loading without hf-hub to mention the missing feature"
    );
}

#[cfg(feature = "local-only")]
#[test]
fn test_from_pretrained_remote_disallowed_by_local_only_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("local-only"),
        "expected remote loading with local-only to mention the local-only restriction"
    );
}
