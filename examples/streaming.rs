//! Synthetic streaming demo: alternates 1 s of band-limited noise with
//! 1 s of silence, then prints every closed segment.

use firered_vad::{Vad, VadEvent};

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
    vad.push_samples(chunk)?;
    while let Some(event) = vad.poll_event() {
      if let VadEvent::SegmentClosed(s) = event {
        println!(
          "segment: {:.3}s..{:.3}s ({} samples)",
          s.start().as_secs_f32(),
          s.end().as_secs_f32(),
          s.sample_count()
        );
      }
    }
  }

  vad.finish()?;
  while let Some(event) = vad.poll_event() {
    if let VadEvent::SegmentClosed(s) = event {
      println!(
        "segment (trailing): {:.3}s..{:.3}s ({} samples)",
        s.start().as_secs_f32(),
        s.end().as_secs_f32(),
        s.sample_count()
      );
    }
  }

  Ok(())
}
