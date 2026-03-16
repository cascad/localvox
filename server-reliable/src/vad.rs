//! VAD wrapper (same semantics as Python vad_util).
//! webrtc-vad: 20ms frame = 320 samples at 16kHz.

use webrtc_vad::{SampleRate, Vad, VadMode};

const FRAME_SAMPLES: usize = 320; // 20 ms at 16 kHz

/// Aggressiveness 1 (LowBitrate). Uses 20ms frames.
pub struct VadDetector {
    vad: Vad,
    frame_bytes: usize,
    silence_threshold_frames: u32,
    silence_frames: u32,
}

impl VadDetector {
    pub fn new(_sample_rate: u32, silence_threshold_frames: u32) -> Self {
        let mut vad = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, VadMode::LowBitrate);
        vad.set_sample_rate(SampleRate::Rate16kHz);
        Self {
            vad,
            frame_bytes: FRAME_SAMPLES * 2,
            silence_threshold_frames: silence_threshold_frames.max(1),
            silence_frames: 0,
        }
    }

    /// Process a chunk. Returns (is_speech, should_flush).
    /// Важно: проверяем ВСЕ 20ms-фреймы в чанке, иначе по первому фрейму решаем о 256ms — теряем середину.
    pub fn process_frame(&mut self, pcm: &[u8]) -> (bool, bool) {
        if pcm.len() < self.frame_bytes {
            return (true, false);
        }
        let mut any_speech = false;
        let mut consecutive_silence = 0u32;
        let mut max_silence = 0u32;
        for frame in pcm.chunks(self.frame_bytes) {
            if frame.len() < self.frame_bytes {
                break;
            }
            let samples: Vec<i16> = frame
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            let is_speech = self.vad.is_voice_segment(&samples).unwrap_or(true);
            if is_speech {
                any_speech = true;
                consecutive_silence = 0;
            } else {
                consecutive_silence += 1;
                max_silence = max_silence.max(consecutive_silence);
            }
        }
        if any_speech {
            self.silence_frames = 0;
            (true, false)
        } else {
            self.silence_frames += max_silence.max(1);
            let should_flush = self.silence_frames >= self.silence_threshold_frames;
            (false, should_flush)
        }
    }

    pub fn reset_silence(&mut self) {
        self.silence_frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_chunk_returns_speech() {
        let mut vad = VadDetector::new(16000, 10);
        let (is_speech, should_flush) = vad.process_frame(&[0u8; 100]);
        assert!(is_speech, "chunk < frame_bytes treated as speech");
        assert!(!should_flush);
    }

    #[test]
    fn test_silence_accumulates() {
        let mut vad = VadDetector::new(16000, 3);
        let silence = vec![0i16; 320];
        let pcm: Vec<u8> = silence.iter().flat_map(|s| s.to_le_bytes()).collect();
        let (s1, f1) = vad.process_frame(&pcm);
        let (s2, f2) = vad.process_frame(&pcm);
        let (s3, f3) = vad.process_frame(&pcm);
        assert!(!s1);
        assert!(!s2);
        assert!(!s3);
        assert!(!f1);
        assert!(!f2);
        assert!(f3, "after 3 silence frames should flush");
    }

    #[test]
    fn test_reset_silence() {
        let mut vad = VadDetector::new(16000, 2);
        let silence: Vec<u8> = vec![0i16; 320]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let _ = vad.process_frame(&silence);
        let _ = vad.process_frame(&silence);
        vad.reset_silence();
        let (_, should_flush) = vad.process_frame(&silence);
        assert!(!should_flush);
    }
}
