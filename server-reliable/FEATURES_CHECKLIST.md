# Реестр фич и логики сервера (чеклист для сверки после рефакторинга)

## 1. STARTUP / main()

- **F1.1** Tracing init с `EnvFilter` (default `info`)
- **F1.2** CLI args: `--host` (default 127.0.0.1), `--port` (default 9745), `--model` (optional override)
- **F1.3** Загрузка `settings.json` через `load_settings()` с multi-path search (`LOCALVOX_SETTINGS`, exe dir, cwd и т.д.)
- **F1.4** `add_cuda_to_path()` — добавление CUDA DLL dirs в PATH: settings.cuda_path + CUDA_PATH env, дедупликация
- **F1.5** Загрузка ASR моделей через `asr::load_models()`, bail если пусто
- **F1.6** `whisper_rs::print_system_info()`
- **F1.7** Лог-таблица: модель / backend (CPU/GPU)
- **F1.8** Опциональное создание `LlmDispatcher` (если `llm_correction_enabled`)
- **F1.9** Создание `AsrDispatcher` с N workers
- **F1.10** TCP listener bind
- **F1.11** Background task: session cleanup каждые 5 мин
- **F1.12** Создание `SessionRegistry`
- **F1.13** `startup_session_scan()` — сканирование audio_dir при старте
- **F1.14** Background task: evict completed sessions каждые 60 сек
- **F1.15** Accept loop: `thread::spawn` + отдельный tokio runtime на каждый коннект

## 2. SESSION MANAGEMENT

- **F2.1** Session ID = UUIDv7 (лексикографически = по времени)
- **F2.2** `DiskSession` struct: dir, session_id, client_id, is_done
- **F2.3** `write_session_meta()` — пишет `session.json` c client_id + created_at
- **F2.4** `mark_session_done()` — создаёт файл `.done` в session dir
- **F2.5** `scan_sessions_on_disk()` — читает `session_*` dirs, парсит session.json, проверяет .done
- **F2.6** `ReadClientId` trait → читает client_id из session.json
- **F2.7** `startup_session_scan()`:
  - Группировка активных сессий по client_id
  - UUIDv7 сортировка: newest = активная, остальные → mark_done
  - Orphan sessions (без client_id) → mark_done
  - Лог-таблица client_id → session_id
- **F2.8** `SessionRegistry`: HashMap<session_id, Session> + HashMap<client_id, session_id>
- **F2.9** `get_active_session_for_client()` — проверяет processing_complete, чистит если complete
- **F2.10** `cancel_live_timeout()` — отмена oneshot таймаута при реконнекте
- **F2.11** `evict_completed()` — удаление старых completed сессий по max_age + cleanup client_sessions
- **F2.12** `client_sessions_snapshot()` — для логирования

## 3. CLIENT CONNECTION (handle_client)

- **F3.1** WebSocket accept + split sender/receiver
- **F3.2** Парсинг первого сообщения: type="config" с client_id, last_seq, source_count, mode
- **F3.3** Resolve session_id: existing active для client_id или новый UUIDv7
- **F3.4** Priority: mode="live" → High, иначе Normal
- **F3.5** SessionMode: High → Live, Normal → Batch

## 4. SESSION RESUME (existing session)

- **F4.1** Если `end_of_stream` — replay completed session и закрыть
- **F4.2** Если Live — cancel timeout, replay transcript, resume AudioWriter
- **F4.3** `replay_completed_session()` — отправляет session state + replay transcript (no_register)
- **F4.4** `OutputSink::replay_to_client()` — replay с skip_to_seq, обновление next_seq
- **F4.5** Resume AudioWriter: scan файлов + transcript.jsonl для max_seq

## 5. NEW SESSION

- **F5.1** AudioWriter: start_session с session_id
- **F5.2** write_session_meta()
- **F5.3** Создание OutputSink с transcript_path
- **F5.4** register_session в AsrDispatcher → SessionHandle
- **F5.5** Создание FileProcessor (thin wrapper)
- **F5.6** Создание Session struct, insert в registry + set_client_session
- **F5.7** Отправка `{"type":"session","state":"new","session_id":...}`

## 6. WS MESSAGE LOOP (run_session_loop)

- **F6.1** Binary: byte[0]=source_id, byte[1..]=PCM → AudioWriter.feed → enqueue completed segments
- **F6.2** source_id: 0 или 1, отбрасывает >1
- **F6.3** Text: handle_text_msg dispatcher

## 7. TEXT MESSAGES (handle_text_msg)

- **F7.1** `recording`: enable/disable recording в AudioWriter
- **F7.2** `config`: set source_count (clamp 1..2)
- **F7.3** `end_of_stream` / `end_session`:
  - Flush writer, enqueue final segments
  - wait_until_empty (3600s)
  - send Done
  - mark processing_complete
  - end_session: mark_session_done + remove_client_session + remove from registry

## 8. DISCONNECT HANDLING (handle_disconnect)

- **F8.1** Если writer_flushed — return
- **F8.2** Close writer, enqueue remaining
- **F8.3** Live mode: spawn timeout task (live_reconnect_timeout_sec)
  - По таймауту: end_of_stream, wait_until_empty, send Done, mark_done, remove_client_session
  - Отмена через oneshot cancel
- **F8.4** Batch mode: spawn blocking thread, аналогичная финализация

## 9. WS HELPERS

- **F9.1** `setup_client_bridge()` — crossbeam→mpsc bridge (thread)
- **F9.2** `spawn_ws_forwarder()` — mpsc→WebSocket (tokio task)
- **F9.3** `msg_to_json()` — ClientMessage → JSON string (Transcript с seq, Status, Done)

## 10. STATUS / LAG POLLING

- **F10.1** `spawn_poll_lag()` — каждые 2 сек
- **F10.2** `scan_session_lag()` — парсинг .meta.json файлов, подсчёт lag по source, размер WAV файлов
- **F10.3** Status JSON: lag_sec, lag_sec_0, lag_sec_1, audio_dir_size_mb, llm_queue, task_queue_size
- **F10.4** `emit_status()` — из worker state: queue_len, pending, worker_busy, last_proc_sec, last_audio_sec

## 11. AUDIO WRITER

- **F11.1** PCM → rotating WAV files (max_chunk_sec, min_chunk_sec)
- **F11.2** VAD-based flushing (vad_silence_sec порог)
- **F11.3** Overlap: хранение последних N сек PCM, prepend в начало нового файла
- **F11.4** .part → .wav rename + .meta.json (start_time_sec, end_time_sec, source_id, duration_sec)
- **F11.5** cumulative_audio_sec tracking для правильных временных меток
- **F11.6** resume_session: scan файлов + transcript.jsonl через `parse_seg_id_into()`
- **F11.7** Multi-source (0..1), source_count clamp 1..2
- **F11.8** set_recording(false) → flush_all

## 12. ASR PIPELINE (dispatcher.rs)

- **F12.1** Priority scheduling: BinaryHeap с High > Normal, FIFO внутри приоритета
- **F12.2** AsrTask: wav_path, source_id, session_id, priority, submitted_at, seq_id
- **F12.3** Multi-worker ASR: N thread workers, condvar wake
- **F12.4** Single model: transcribe → filter_hallucinations
- **F12.5** Multi model: parallel threads per model → collect outputs
- **F12.6** LLM path: ensemble_merge_n → merged text для LLM
- **F12.7** Raw path (no LLM): pick first non-empty, or variants
- **F12.8** Per-session ordered finalization: BTreeMap[source_id] с next_seq
- **F12.9** drain_finalize: pop in order, skip None (no speech)
- **F12.10** Cleanup: remove WAV + meta.json после обработки
- **F12.11** enqueue_seq: AtomicU64 per source для монотонной нумерации
- **F12.12** pending_count tracking + llm_inflight tracking
- **F12.13** wait_session_empty: poll pending_count + llm_inflight

## 13. LLM CORRECTION (dispatcher.rs + llm_corrector.rs)

- **F13.1** LLM path в drain_finalize: build AsrResult, context_snapshot, prev_tail
- **F13.2** context_lines: VecDeque, last 5 raw ASR lines per source
- **F13.3** AsrResult: сериализуется в .asr.json на диск
- **F13.4** LlmDispatcher: BinaryHeap priority queue + N worker threads
- **F13.5** LlmCorrector::correct(): single vs ensemble prompt
- **F13.6** Prompt templates: {context_str}, {merged_text}, {model_variants}, {whisper_text}, {gigaam_text}
- **F13.7** Dynamic num_predict: merged_text.len().max(96).min(384)
- **F13.8** Stop sequences: "А.Семкин", "Корректор А.Егорова", "корректор А.Егорова", "редактор субтитров"
- **F13.9** Temperature 0.1
- **F13.10** Thinking model detection: "3.5" или "deepseek" → think: false
- **F13.11** Length guard: response > 3x merged → truncate_at_sentence_boundary → если нет — discard
- **F13.12** strip_llm_reasoning: удаление мета-префиксов ("Используем вариант", "Corrected:" и т.д.)
- **F13.13** Strip wrapping quotes: «» "" ""
- **F13.14** After LLM: merge_overlap(prev_tail, corrected) → emit + update prev_tail
- **F13.15** Cleanup: remove WAV + meta.json + asr.json после LLM

## 14. OUTPUT SINK (processor.rs)

- **F14.1** Dual-write: transcript.jsonl (append) + crossbeam channel (WebSocket)
- **F14.2** Monotonic seq via AtomicU64 (fetch_add)
- **F14.3** Transcript JSON: seq, text, source, start_sec, end_sec, seg_id, variants
- **F14.4** Done: `{"type":"done"}` в jsonl
- **F14.5** send_status: только channel, не файл
- **F14.6** clear_client: None → channel
- **F14.7** replay_to_client: parse jsonl, skip seq ≤ skip_to_seq, restore next_seq from max_seq
- **F14.8** replay_to_client_no_register: то же но без register sender

## 15. ASR MODELS (asr/)

- **F15.1** AsrModel trait: name(), backend(), transcribe(), filter_hallucinations()
- **F15.2** WhisperAdapter: whisper-rs, CUDA/CPU, no_speech_thold
- **F15.3** GigaAmAdapter: sherpa-rs (ONNX), CUDA/CPU
- **F15.4** ParakeetAdapter: sherpa-rs (TDT), CUDA/CPU
- **F15.5** SileroAdapter: ort (ONNX Runtime)
- **F15.6** load_models: runtime dispatch по model_type string

## 16. HALLUCINATION FILTER (hallucination.rs)

- **F16.1** Per-model drop/strip lists (Whisper, GigaAM, Silero, Parakeet)
- **F16.2** Whisper drops: субтитры, музыка, звуки, subscribe, www и т.д.
- **F16.3** Strip embedded phrases: "Субтитры создал X", "Редактор субтитров...", "А.Семкин, корректор А.Егорова"
- **F16.4** Bracket-dominated text filter: [музыка] * и т.д. → пусто
- **F16.5** Parakeet drops: "yeah", "pinch"
- **F16.6** Drop logic: pattern + word_count proximity check (не дропает если текст значительно длиннее паттерна)

## 17. TEXT POST-PROCESSING

- **F17.1** merge_overlap: word-level overlap dedup (TAIL_WORDS=15)
- **F17.2** Partial word overlap: "оф" + "офис" → "ис"
- **F17.3** Case-insensitive + punctuation-ignoring match
- **F17.4** ensemble_merge_n: pairwise word-level Levenshtein alignment
- **F17.5** pick_best_word: prefer longer word on conflict

## 18. SESSION CLEANUP (session_cleanup.rs)

- **F18.1** TTL-based: session_ttl_hours
- **F18.2** Size-based: audio_dir_max_mb
- **F18.3** .done sessions: DONE_SESSION_TTL = 5 min (без grace)
- **F18.4** Grace period: grace_minutes для активных (не .done) сессий
- **F18.5** Sorted by mtime, удаление oldest first

## 19. VAD (vad.rs)

- **F19.1** webrtc-vad, Mode=LowBitrate, 16kHz
- **F19.2** 20ms frames (320 samples)
- **F19.3** Проверка ВСЕХ фреймов в чанке (не только первого)
- **F19.4** Silence accumulation + threshold → should_flush

## 20. CONFIG (config.rs)

- **F20.1** 25+ настроек с defaults
- **F20.2** Legacy model_path → resolved_models() с ensemble
- **F20.3** New models array: type/model_path/tokens_path/use_gpu
- **F20.4** Multi-path settings search

## 21. TESTS

- **F21.1** config: 5 tests (defaults, parse, minimal, resolved_models legacy/new)
- **F21.2** vad: 3 tests (small chunk, silence, reset)
- **F21.3** overlap: 7 tests (empty, exact, partial, no overlap)
- **F21.4** hallucination: 10 tests (drop, strip, keep, brackets)
- **F21.5** transcript_postprocess: 12 tests (ensemble merge, process_segment)
- **F21.6** llm_corrector: 7 tests (strip reasoning, quotes, passthrough)
