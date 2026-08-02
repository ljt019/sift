#![cfg(feature = "rocm")]

use anyhow::Result;
use sift_embedding_runtime::nn::{ops::softmax_last_dim, rotary_emb::rope};
use sift_embedding_runtime::{D, DType, Device, Tensor};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

fn values(count: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let centered = (index * multiplier % modulus) as f32 - (modulus / 2) as f32;
            centered * scale
        })
        .collect()
}

fn linear3(input: &Tensor, weight: &Tensor, output_size: usize) -> Result<Tensor> {
    let (batch, tokens, input_size) = input.dims3()?;
    Ok(input
        .reshape((batch * tokens, input_size))?
        .matmul(&weight.t()?)?
        .reshape((batch, tokens, output_size))?)
}

fn repeat_key_values(input: &Tensor, groups: usize) -> Result<Tensor> {
    let (batch, heads, tokens, dimension) = input.dims4()?;
    Ok(input
        .unsqueeze(2)?
        .expand((batch, heads, groups, tokens, dimension))?
        .reshape((batch, heads * groups, tokens, dimension))?)
}

fn model_shaped_pipeline(device: &Device) -> Result<Vec<f32>> {
    const BATCH: usize = 2;
    const TOKENS: usize = 17;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const HIDDEN: usize = HEADS * HEAD_DIM;
    const INTERMEDIATE: usize = 48;

    let input = Tensor::from_vec(
        values(BATCH * TOKENS * HIDDEN, 17, 43, 0.015),
        (BATCH, TOKENS, HIDDEN),
        device,
    )?;
    let q_weight = Tensor::from_vec(
        values(HIDDEN * HIDDEN, 13, 37, 0.007),
        (HIDDEN, HIDDEN),
        device,
    )?;
    let kv_weight = Tensor::from_vec(
        values(KV_HEADS * HEAD_DIM * HIDDEN, 19, 41, 0.006),
        (KV_HEADS * HEAD_DIM, HIDDEN),
        device,
    )?;
    let output_weight = Tensor::from_vec(
        values(HIDDEN * HIDDEN, 23, 47, 0.005),
        (HIDDEN, HIDDEN),
        device,
    )?;

    let query = linear3(&input, &q_weight, HIDDEN)?
        .reshape((BATCH, TOKENS, HEADS, HEAD_DIM))?
        .transpose(1, 2)?;
    let key = linear3(&input, &kv_weight, KV_HEADS * HEAD_DIM)?
        .reshape((BATCH, TOKENS, KV_HEADS, HEAD_DIM))?
        .transpose(1, 2)?;
    let value = key.clone();

    let positions = Tensor::arange(0_u32, TOKENS as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((TOKENS, 1))?;
    let inverse_frequencies = Tensor::from_slice(&[1.0_f32, 0.1, 0.01, 0.001], (1, 4), device)?;
    let frequencies = positions.matmul(&inverse_frequencies)?;
    let cosine = frequencies.cos()?.contiguous()?;
    let sine = frequencies.sin()?.contiguous()?;
    let query = rope(&query.contiguous()?, &cosine, &sine)?;
    let key = repeat_key_values(&rope(&key.contiguous()?, &cosine, &sine)?, HEADS / KV_HEADS)?;
    let value = repeat_key_values(&value, HEADS / KV_HEADS)?;

    let attention_bias = Tensor::from_vec(
        (0..BATCH * TOKENS)
            .map(|index| {
                let token = index % TOKENS;
                if index / TOKENS == 1 && token >= 13 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
        (BATCH, 1, 1, TOKENS),
        device,
    )?;
    let scores = (query.matmul(&key.transpose(2, 3)?)? * (1.0 / 4.0_f64.sqrt()))?
        .broadcast_add(&attention_bias)?;
    let context = softmax_last_dim(&scores)?
        .matmul(&value)?
        .transpose(1, 2)?
        .reshape((BATCH, TOKENS, HIDDEN))?;
    let projected = linear3(&context, &output_weight, HIDDEN)?;

    let norm_weight = Tensor::from_vec(values(HIDDEN, 7, 29, 0.002), HIDDEN, device)?;
    let normalized = projected.broadcast_div(
        &((projected.sqr()?.sum_keepdim(D::Minus1)? / HIDDEN as f64)? + 1e-6)?.sqrt()?,
    )?;
    let normalized = normalized.broadcast_mul(&(&norm_weight + 1.0)?)?;

    let gate_weight = Tensor::from_vec(
        values(INTERMEDIATE * HIDDEN, 29, 53, 0.004),
        (INTERMEDIATE, HIDDEN),
        device,
    )?;
    let up_weight = Tensor::from_vec(
        values(INTERMEDIATE * HIDDEN, 31, 59, 0.004),
        (INTERMEDIATE, HIDDEN),
        device,
    )?;
    let down_weight = Tensor::from_vec(
        values(HIDDEN * INTERMEDIATE, 37, 61, 0.004),
        (HIDDEN, INTERMEDIATE),
        device,
    )?;
    let gate = linear3(&normalized, &gate_weight, INTERMEDIATE)?.gelu()?;
    let up = linear3(&normalized, &up_weight, INTERMEDIATE)?;
    let hidden = (&normalized + &linear3(&(gate * up)?, &down_weight, HIDDEN)?)?;

    let pooling_mask = Tensor::from_vec(
        (0..BATCH * TOKENS)
            .map(|index| {
                if index / TOKENS == 1 && index % TOKENS >= 13 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>(),
        (BATCH, TOKENS, 1),
        device,
    )?;
    let token_counts = Tensor::from_slice(&[TOKENS as f32, 13.0], (BATCH, 1), device)?;
    let pooled = hidden
        .broadcast_mul(&pooling_mask)?
        .sum(1)?
        .broadcast_div(&token_counts)?
        .narrow(D::Minus1, 0, 16)?;
    let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(pooled
        .broadcast_div(&norm)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

#[test]
#[ignore = "requires a working ROCm device and runtime"]
fn rocm_tensor_primitives_match_expected_values() -> Result<()> {
    let invalid = Device::new_rocm(usize::MAX)
        .expect_err("an impossible ROCm ordinal must be rejected")
        .to_string();
    assert!(invalid.contains("ROCm device ordinal") && invalid.contains("out of range"));

    let device = Device::new_rocm(0)?;

    assert!(device.begin_memory_profile()?);
    let profiled = Tensor::from_vec(vec![0.0_f32; 1024], 1024, &device)?;
    let profile = device
        .end_memory_profile()?
        .expect("ROCm devices support memory profiling");
    assert!(profile.free_bytes > 0);
    drop(profiled);

    let left = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;
    let right = Tensor::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), &device)?;
    let product = left.matmul(&right)?.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&product, &[58.0, 64.0, 139.0, 154.0], 1e-5);

    let ids = Tensor::from_slice(&[2_u32, 0], 2, &device)?;
    let selected = left
        .index_select(&ids, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&selected, &[3.0, 1.0, 6.0, 4.0], 0.0);
    assert_close(
        &ids.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        &[2.0, 0.0],
        0.0,
    );

    let scales = Tensor::from_slice(&[10.0_f32, 100.0, 1000.0], (1, 3), &device)?;
    let broadcast = left
        .broadcast_mul(&scales)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&broadcast, &[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0], 0.0);

    let transposed = left.t()?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&transposed, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0.0);

    let attention_input = Tensor::from_slice(
        &[
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        (1, 2, 2, 3),
        &device,
    )?;
    let attention_scores = attention_input
        .matmul(&attention_input.transpose(2, 3)?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(
        &attention_scores,
        &[14.0, 32.0, 32.0, 77.0, 194.0, 266.0, 266.0, 365.0],
        1e-5,
    );

    let transformed = ((&left * 2.0)? + 1.0)?.sqrt()?;
    let sums = transformed.sum(D::Minus1)?.to_vec1::<f32>()?;
    assert_close(
        &sums,
        &[
            3.0_f32.sqrt() + 5.0_f32.sqrt() + 7.0_f32.sqrt(),
            9.0_f32.sqrt() + 11.0_f32.sqrt() + 13.0_f32.sqrt(),
        ],
        1e-5,
    );

    let logits = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, -1.0, 0.0, 1.0], (2, 3), &device)?;
    let probabilities = softmax_last_dim(&logits)?.flatten_all()?.to_vec1::<f32>()?;
    assert_close(
        &probabilities,
        &[
            0.09003057, 0.24472848, 0.66524094, 0.09003057, 0.24472848, 0.66524094,
        ],
        1e-6,
    );

    let wide_logits = values(2 * 1537, 41, 101, 0.025);
    let cpu_probabilities =
        softmax_last_dim(&Tensor::from_slice(&wide_logits, (2, 1537), &Device::Cpu)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
    let rocm_probabilities =
        softmax_last_dim(&Tensor::from_slice(&wide_logits, (2, 1537), &device)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
    assert_close(&rocm_probabilities, &cpu_probabilities, 2e-6);

    let wide_reduction = values(3 * 2053, 43, 97, 0.001);
    let cpu_sums = Tensor::from_slice(&wide_reduction, (3, 2053), &Device::Cpu)?
        .sum(D::Minus1)?
        .to_vec1::<f32>()?;
    let rocm_sums = Tensor::from_slice(&wide_reduction, (3, 2053), &device)?
        .sum(D::Minus1)?
        .to_vec1::<f32>()?;
    assert_close(&rocm_sums, &cpu_sums, 1e-4);

    let source = Tensor::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (1, 1, 2, 4),
        &device,
    )?;
    let cosine = Tensor::from_slice(&[1.0_f32, 1.0, 0.0, 0.0], (2, 2), &device)?;
    let sine = Tensor::from_slice(&[0.0_f32, 0.0, 1.0, 1.0], (2, 2), &device)?;
    let rotated = rope(&source, &cosine, &sine)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&rotated, &[1.0, 2.0, 3.0, 4.0, -7.0, -8.0, 5.0, 6.0], 0.0);

    Ok(())
}

#[test]
#[ignore = "requires a working ROCm device and runtime"]
fn rocm_model_shaped_pipeline_matches_cpu_and_is_repeatable() -> Result<()> {
    let cpu = model_shaped_pipeline(&Device::Cpu)?;
    let device = Device::new_rocm(0)?;
    let first = model_shaped_pipeline(&device)?;
    let second = model_shaped_pipeline(&device)?;

    assert!(first.iter().all(|value| value.is_finite()));
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "repeated ROCm execution changed output bits"
    );

    let (maximum, squared_error, dot, cpu_norm, rocm_norm) = cpu.iter().zip(&first).fold(
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
        |(maximum, squared_error, dot, cpu_norm, rocm_norm), (&cpu, &rocm)| {
            let error = (cpu - rocm).abs();
            (
                maximum.max(error),
                squared_error + error * error,
                dot + cpu * rocm,
                cpu_norm + cpu * cpu,
                rocm_norm + rocm * rocm,
            )
        },
    );
    let rmse = (squared_error / cpu.len() as f32).sqrt();
    let cosine = dot / (cpu_norm * rocm_norm).sqrt();
    assert!(
        maximum < 2e-5 && rmse < 3e-6 && cosine > 0.999_999,
        "model-shaped CPU/ROCm mismatch: cosine={cosine}, max_abs={maximum:e}, rmse={rmse:e}"
    );
    println!(
        "ROCM_PIPELINE=PASS repeat_exact=true cosine={cosine} max_abs={maximum:e} rmse={rmse:e}"
    );

    Ok(())
}
