.PHONY: fmt fmt-check test clippy ci

# Применить форматирование
fmt:
	cargo fmt --all

# Проверить форматирование (как в CI)
fmt-check:
	cargo fmt --all -- --check

# Тесты клиента и сервера (сервер — CPU-only, как в CI)
test:
	cargo test -p live-transcribe-client-reliable
	cargo test -p live-transcribe-server-reliable --no-default-features

# Clippy для клиента и сервера
clippy:
	cargo clippy -p live-transcribe-client-reliable -- -D warnings
	cargo clippy -p live-transcribe-server-reliable --no-default-features -- -D warnings

# Всё как в CI: fmt-check, test, clippy
ci: fmt-check test clippy
