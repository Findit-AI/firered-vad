//! Synthetic streaming demo: alternates 1 s of band-limited noise with
//! 1 s of silence, then prints every closed segment.

use firered_vad::Vad;

const SAMPLE_RATE_HZ: u32 = 16_000;

fn band_noise(duration_secs: f32) -> Vec<f32> {
  let n = (duration_secs * SAMPLE_RATE_HZ as f32) as usize;
  let mut buf = Vec::with_capacity(n);
  let formants = [200.0f32, 700.0, 1_500.0, 2_500.0, 3_500.0];
  for i in 0..n {
    let t = i as f32 / SAMPLE_RATE_HZ as f32;
    let mut sample = 0.0f32;
    for f in formants {
      sample += (core::f32::consts::TAU * f * t).sin();
    }
    buf.push(0.1 * sample);
  }
  buf
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut vad = Vad::bundled()?;

  let mut pcm = Vec::new();
  for _ in 0..3 {
    pcm.extend(band_noise(1.0));
    pcm.extend(vec![0.0; SAMPLE_RATE_HZ as usize]);
  }

  for chunk in pcm.chunks(1_600) {
    let mut chunk: &[f32] = chunk;
    while let Some(segment) = vad.push_samples(chunk)? {
      println!(
        "segment: {:.3}s..{:.3}s ({} samples)",
        segment.start().as_secs_f32(),
        segment.end().as_secs_f32(),
        segment.sample_count()
      );
      chunk = &[]; // drain remaining buffered segments before pushing the next chunk
    }
  }

  if let Some(segment) = vad.finish()? {
    println!(
      "segment (trailing): {:.3}s..{:.3}s ({} samples)",
      segment.start().as_secs_f32(),
      segment.end().as_secs_f32(),
      segment.sample_count()
    );
  }

  Ok(())
}
