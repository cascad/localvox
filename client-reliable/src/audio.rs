//! Audio capture: microphones, loopback (system sound), device resolution.

#[cfg(windows)]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::KeyCode;

use crate::types::{ServerMessage, UiEvent, WsOutgoing};

pub const SAMPLE_RATE: u32 = 16000;
pub const CHUNK_FRAMES: usize = 512;

/// Физическая клавиша (не зависит от раскладки). Русская ЙЦУКЕН → Latin QWERTY.
pub fn physical_key(c: char) -> char {
    match c {
        'q' | 'Q' | 'й' | 'Й' => 'q',
        'w' | 'W' | 'ц' | 'Ц' => 'w',
        'e' | 'E' | 'у' | 'У' => 'e',
        'r' | 'R' | 'к' | 'К' => 'r',
        't' | 'T' | 'е' | 'Е' => 't',
        'y' | 'Y' | 'н' | 'Н' => 'y',
        'u' | 'U' | 'г' | 'Г' => 'u',
        'i' | 'I' | 'ш' | 'Ш' => 'i',
        'o' | 'O' | 'щ' | 'Щ' => 'o',
        'p' | 'P' | 'з' | 'З' => 'p',
        'a' | 'A' | 'ф' | 'Ф' => 'a',
        's' | 'S' | 'ы' | 'Ы' => 's',
        'd' | 'D' | 'в' | 'В' => 'd',
        'f' | 'F' | 'а' | 'А' => 'f',
        'g' | 'G' | 'п' | 'П' => 'g',
        'h' | 'H' | 'р' | 'Р' => 'h',
        'j' | 'J' | 'о' | 'О' => 'j',
        'k' | 'K' | 'л' | 'Л' => 'k',
        'l' | 'L' | 'д' | 'Д' => 'l',
        'z' | 'Z' | 'я' | 'Я' => 'z',
        'x' | 'X' | 'ч' | 'Ч' => 'x',
        'c' | 'C' | 'с' | 'С' => 'c',
        'v' | 'V' | 'м' | 'М' => 'v',
        'b' | 'B' | 'и' | 'И' => 'b',
        'n' | 'N' | 'т' | 'Т' => 'n',
        'm' | 'M' | 'ь' | 'Ь' => 'm',
        _ => c.to_ascii_lowercase(),
    }
}

pub fn key_matches(key: KeyCode, expected: char) -> bool {
    match key {
        KeyCode::Char(c) => physical_key(c) == expected,
        _ => false,
    }
}

/// Реальные микрофоны (без cpal loopback aggregate на macOS).
pub fn collect_input_devices() -> Vec<(cpal::Device, String)> {
    let host = cpal::default_host();
    let mut list = Vec::new();
    for dev in host.input_devices().unwrap_or_else(|_| panic!("input_devices")) {
        let name = dev.name().unwrap_or_default();
        #[cfg(target_os = "macos")]
        if name.contains("Cpal loopback") || name.contains("cpal output recorder") {
            continue;
        }
        list.push((dev, name));
    }
    list
}

/// Строка для отображения устройства (имя + id + производитель — чтобы различать дубликаты).
pub fn format_device_display(dev: &cpal::Device, name: &str, extra: &str) -> String {
    let id_str = dev.id().map(|id| format!("{id}")).unwrap_or_else(|_| "?".into());
    let mut parts = vec![name.to_string()];
    if let Ok(desc) = dev.description() {
        if let Some(mfr) = desc.manufacturer() {
            let mfr = mfr.trim();
            if !mfr.is_empty() && mfr != name {
                parts.push(mfr.to_string());
            }
        }
    }
    parts.push(format!("id:{id_str}"));
    if !extra.is_empty() {
        parts.push(extra.to_string());
    }
    parts.join(" | ")
}

/// Устройства для loopback (системный звук).
pub fn list_output_device_names() -> Vec<(usize, String)> {
    #[cfg(windows)]
    {
        let enumerator = match wasapi::DeviceEnumerator::new() {
            Ok(e) => e,
            Err(_) => return vec![],
        };
        let collection = match enumerator.get_device_collection(&wasapi::Direction::Render) {
            Ok(c) => c,
            Err(_) => return vec![],
        };
        let mut list: Vec<(usize, String)> = collection
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| {
                let dev = r.ok()?;
                dev.get_friendlyname().ok().map(|n| (i + 1, n))
            })
            .collect();
        list.insert(0, (0, "default-output".to_string()));
        list
    }

    #[cfg(target_os = "macos")]
    {
        let devices = match cpal::default_host().output_devices() {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        let mut list: Vec<(usize, String)> = devices
            .enumerate()
            .filter_map(|(i, dev)| dev.name().ok().map(|n| (i + 1, n)))
            .collect();
        list.insert(0, (0, "default-output".to_string()));
        list
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        let devices = match cpal::default_host().input_devices() {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        devices
            .enumerate()
            .filter_map(|(i, dev)| dev.name().ok().map(|n| (i, n)))
            .collect()
    }
}

pub fn resolve_device(query: &str) -> Result<cpal::Device> {
    let devices = collect_input_devices();

    if let Ok(idx) = query.parse::<usize>() {
        return devices
            .into_iter()
            .nth(idx)
            .map(|(dev, _)| dev)
            .context(format!("Устройство с индексом {idx} не найдено"));
    }

    let needle = query.to_lowercase();
    let matches: Vec<(usize, cpal::Device, String)> = devices
        .into_iter()
        .enumerate()
        .filter_map(|(i, (dev, name))| {
            if name.to_lowercase().contains(&needle) {
                Some((i, dev, name))
            } else {
                None
            }
        })
        .collect();

    match matches.len() {
        0 => anyhow::bail!("Устройство «{query}» не найдено. Запустите --list-devices"),
        1 => Ok(matches.into_iter().next().unwrap().1),
        _ => {
            eprintln!("⚠ «{query}» совпало с несколькими устройствами, выбрано первое:");
            for (i, _, name) in &matches {
                eprintln!("  [{i}] {name}");
            }
            Ok(matches.into_iter().next().unwrap().1)
        }
    }
}

pub fn default_device() -> Result<cpal::Device> {
    cpal::default_host()
        .default_input_device()
        .context("Нет устройства ввода по умолчанию")
}

pub fn to_mono(data: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

pub fn resample(src: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE {
        return src.to_vec();
    }
    let ratio = src_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (src.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;
        let a = src[idx.min(src.len() - 1)];
        let b = src[(idx + 1).min(src.len() - 1)];
        out.push(a + (b - a) * frac as f32);
    }
    out
}

pub fn audio_capture(
    device: cpal::Device,
    source_id: u8,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    let supported = if device.supports_input() {
        device.default_input_config()?
    } else {
        match device.default_output_config() {
            Ok(c) => c,
            Err(cpal::DefaultStreamConfigError::StreamTypeNotSupported) => {
                device
                    .supported_output_configs()
                    .context("supported_output_configs")?
                    .next()
                    .context("Нет supported output configs")?
                    .with_max_sample_rate()
            }
            Err(e) => return Err(e.into()),
        }
    };
    let native_rate = supported.sample_rate();
    let native_channels = supported.channels();

    let config = cpal::StreamConfig {
        channels: native_channels,
        sample_rate: native_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    let ui_tx2 = ui_tx.clone();
    let running2 = running.clone();
    let mut pcm_buf: Vec<i16> = Vec::with_capacity(CHUNK_FRAMES * 4);

    let ui_tx_err = ui_tx.clone();
    let src_id = source_id;
    let err_fn = move |err: cpal::StreamError| {
        let _ = ui_tx_err.send(UiEvent::Server(ServerMessage::Error {
            text: format!("audio src{} stream error: {err}", src_id),
        }));
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !running2.load(Ordering::Relaxed) {
                return;
            }
            let mono = to_mono(data, native_channels);
            let resampled = resample(&mono, native_rate);
            let samples: Vec<i16> = resampled
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            pcm_buf.extend_from_slice(&samples);

            while pcm_buf.len() >= CHUNK_FRAMES {
                let chunk: Vec<i16> = pcm_buf.drain(..CHUNK_FRAMES).collect();
                let rms: f64 =
                    (chunk.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / chunk.len() as f64)
                        .sqrt();
                let level = (rms / 32768.0 * 12.0).min(1.0) as f32;
                let _ = ui_tx2.send(UiEvent::AudioLevel { source: source_id, level });

                let mut bytes = Vec::with_capacity(1 + chunk.len() * 2);
                bytes.push(source_id);
                bytes.extend(chunk.iter().flat_map(|s| s.to_le_bytes()));
                let _ = ws_tx.send(WsOutgoing::Binary(bytes));
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    while running.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }

    drop(stream);
    Ok(())
}

#[cfg(windows)]
fn resolve_output_device(query: &str) -> Result<wasapi::Device> {
    let enumerator = wasapi::DeviceEnumerator::new()
        .map_err(|e| anyhow::anyhow!("DeviceEnumerator: {e:?}"))?;

    if query.eq_ignore_ascii_case("default-output") || query.eq_ignore_ascii_case("default") {
        return enumerator
            .get_default_device(&wasapi::Direction::Render)
            .map_err(|e| anyhow::anyhow!("get_default_device: {e:?}"));
    }

    if let Ok(idx) = query.parse::<usize>() {
        if idx == 0 {
            return enumerator
                .get_default_device(&wasapi::Direction::Render)
                .map_err(|e| anyhow::anyhow!("get_default_device: {e:?}"));
        }
        let collection = enumerator
            .get_device_collection(&wasapi::Direction::Render)
            .map_err(|e| anyhow::anyhow!("get_device_collection: {e:?}"))?;
        return collection
            .into_iter()
            .nth(idx - 1)
            .context(format!("Выходное устройство с индексом {idx} не найдено"))?
            .map_err(|e| anyhow::anyhow!("device error: {e:?}"));
    }

    let needle = query.to_lowercase();
    let collection = enumerator
        .get_device_collection(&wasapi::Direction::Render)
        .map_err(|e| anyhow::anyhow!("get_device_collection: {e:?}"))?;
    for dev_result in collection.into_iter() {
        let dev = dev_result.map_err(|e| anyhow::anyhow!("device error: {e:?}"))?;
        let name = dev.get_friendlyname().unwrap_or_default();
        if name.to_lowercase().contains(&needle) {
            return Ok(dev);
        }
    }

    anyhow::bail!("Выходное устройство «{query}» не найдено. Запустите --list-devices")
}

#[cfg(target_os = "macos")]
fn resolve_output_device_macos(query: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();

    if query.eq_ignore_ascii_case("default-output") || query.eq_ignore_ascii_case("default") {
        return host
            .default_output_device()
            .context("Нет устройства вывода по умолчанию");
    }

    if let Ok(idx) = query.parse::<usize>() {
        if idx == 0 {
            return host
                .default_output_device()
                .context("Нет устройства вывода по умолчанию");
        }
        let devices: Vec<_> = host.output_devices()?.collect();
        return devices
            .into_iter()
            .nth(idx - 1)
            .context(format!("Выходное устройство с индексом {idx} не найдено"));
    }

    let needle = query.to_lowercase();
    for dev in host.output_devices()? {
        let name = dev.name().unwrap_or_default();
        if name.to_lowercase().contains(&needle) {
            return Ok(dev);
        }
    }

    anyhow::bail!("Выходное устройство «{query}» не найдено. Запустите --list-devices")
}

pub fn loopback_capture(
    device_query: String,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    #[cfg(windows)]
    {
        loopback_capture_wasapi(device_query, ws_tx, ui_tx, running)
    }

    #[cfg(target_os = "macos")]
    {
        loopback_capture_macos(device_query, ws_tx, ui_tx, running)
    }

    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        let device = resolve_device(&device_query)?;
        audio_capture(device, 1, ws_tx, ui_tx, running)
    }
}

#[cfg(target_os = "macos")]
fn loopback_capture_macos(
    device_query: String,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    let device = resolve_output_device_macos(&device_query)?;
    audio_capture(device, 1, ws_tx, ui_tx, running)
}

#[cfg(windows)]
fn loopback_capture_wasapi(
    device_query: String,
    ws_tx: mpsc::Sender<WsOutgoing>,
    ui_tx: mpsc::Sender<UiEvent>,
    running: std::sync::Arc<AtomicBool>,
) -> Result<()> {
    wasapi::initialize_mta()
        .ok()
        .context("COM init failed in loopback thread")?;

    let device = resolve_output_device(&device_query)?;
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|e| anyhow::anyhow!("get_iaudioclient: {e:?}"))?;

    let desired_format =
        wasapi::WaveFormat::new(16, 16, &wasapi::SampleType::Int, SAMPLE_RATE as usize, 1, None);

    let (_, min_time) = audio_client
        .get_device_period()
        .map_err(|e| anyhow::anyhow!("get_device_period: {e:?}"))?;

    let mode = wasapi::StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_time,
    };

    audio_client
        .initialize_client(&desired_format, &wasapi::Direction::Capture, &mode)
        .map_err(|e| anyhow::anyhow!("initialize_client loopback: {e:?}"))?;

    let h_event = audio_client
        .set_get_eventhandle()
        .map_err(|e| anyhow::anyhow!("set_get_eventhandle: {e:?}"))?;

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| anyhow::anyhow!("get_audiocaptureclient: {e:?}"))?;

    audio_client
        .start_stream()
        .map_err(|e| anyhow::anyhow!("start_stream: {e:?}"))?;

    let blockalign = desired_format.get_blockalign() as usize;
    let chunk_bytes = CHUNK_FRAMES * blockalign;
    let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(chunk_bytes * 8);

    while running.load(Ordering::Relaxed) {
        capture_client
            .read_from_device_to_deque(&mut sample_queue)
            .map_err(|e| anyhow::anyhow!("read_from_device: {e:?}"))?;

        while sample_queue.len() >= chunk_bytes {
            let mut tagged = Vec::with_capacity(1 + chunk_bytes);
            tagged.push(1u8);
            let mut rms_sum: f64 = 0.0;
            for i in 0..chunk_bytes {
                let b = sample_queue.pop_front().unwrap();
                tagged.push(b);
                if i % 2 == 0 && i + 1 < chunk_bytes {
                    let lo = b as u8;
                    let hi = *sample_queue.front().unwrap_or(&0);
                    let sample = i16::from_le_bytes([lo, hi]);
                    rms_sum += (sample as f64).powi(2);
                }
            }
            let rms = (rms_sum / CHUNK_FRAMES as f64).sqrt();
            let level = (rms / 32768.0 * 12.0).min(1.0) as f32;
            let _ = ui_tx.send(UiEvent::AudioLevel { source: 1, level });
            let _ = ws_tx.send(WsOutgoing::Binary(tagged));
        }

        if h_event.wait_for_event(100).is_err() {}
    }

    audio_client
        .stop_stream()
        .map_err(|e| anyhow::anyhow!("stop_stream: {e:?}"))?;
    Ok(())
}
