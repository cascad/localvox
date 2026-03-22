//! TLS setup for WSS support.

use anyhow::{Context, Result};
use rustls::ServerConfig;
use std::path::Path;
use std::sync::Arc;
use tokio_rustls::TlsAcceptor;

/// Load TLS acceptor from PEM cert and key files.
pub fn load_tls_acceptor(cert_path: &Path, key_path: &Path) -> Result<TlsAcceptor> {
    let certs = rustls_pemfile::certs(&mut std::io::BufReader::new(
        std::fs::File::open(cert_path)
            .with_context(|| format!("open cert {}", cert_path.display()))?,
    ))
    .collect::<Result<Vec<_>, _>>()
    .with_context(|| format!("parse cert {}", cert_path.display()))?;

    let key = rustls_pemfile::private_key(&mut std::io::BufReader::new(
        std::fs::File::open(key_path)
            .with_context(|| format!("open key {}", key_path.display()))?,
    ))
    .with_context(|| format!("parse key {}", key_path.display()))?
    .context("no private key found in file")?;

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("build TLS config")?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}
