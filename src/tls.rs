//! Self-signed TLS material for shared llama-server.

use std::{
    fs,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use rcgen::{CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair, SanType};

pub const CERT_FILE_NAME: &str = "cert.pem";
pub const KEY_FILE_NAME: &str = "key.pem";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsPaths {
    pub cert_file: PathBuf,
    pub key_file: PathBuf,
}

impl TlsPaths {
    pub fn in_config_dir(config_dir: &Path) -> Self {
        let dir = config_dir.join("tls");
        Self {
            cert_file: dir.join(CERT_FILE_NAME),
            key_file: dir.join(KEY_FILE_NAME),
        }
    }
}

/// Ensure a self-signed cert/key pair exists under `{config_dir}/tls/`.
///
/// Regenerates when missing. SANs cover loopback plus any extra IPs (share bind).
pub fn ensure_self_signed(config_dir: &Path, extra_ips: &[IpAddr]) -> Result<TlsPaths> {
    let paths = TlsPaths::in_config_dir(config_dir);
    if paths.cert_file.is_file() && paths.key_file.is_file() {
        return Ok(paths);
    }
    generate_self_signed(&paths, extra_ips)?;
    Ok(paths)
}

fn generate_self_signed(paths: &TlsPaths, extra_ips: &[IpAddr]) -> Result<()> {
    let dir = paths
        .cert_file
        .parent()
        .context("tls cert path has no parent directory")?;
    fs::create_dir_all(dir).with_context(|| format!("could not create {}", dir.display()))?;

    let mut params = CertificateParams::new(vec!["localhost".into()])
        .context("could not build certificate parameters")?;
    params
        .distinguished_name
        .push(DnType::CommonName, "tinyinference");
    params.is_ca = IsCa::NoCa;
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];

    let mut sans = vec![
        SanType::DnsName(
            "localhost"
                .try_into()
                .map_err(|error| anyhow::anyhow!("invalid DNS SAN: {error}"))?,
        ),
        SanType::IpAddress(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        SanType::IpAddress(IpAddr::V6(Ipv6Addr::LOCALHOST)),
    ];
    for ip in extra_ips {
        if *ip == IpAddr::V4(Ipv4Addr::UNSPECIFIED) || *ip == IpAddr::V6(Ipv6Addr::UNSPECIFIED) {
            continue;
        }
        if !sans.iter().any(|san| matches!(san, SanType::IpAddress(existing) if existing == ip)) {
            sans.push(SanType::IpAddress(*ip));
        }
    }
    params.subject_alt_names = sans;

    let key_pair = KeyPair::generate().context("could not generate TLS key")?;
    let cert = params
        .self_signed(&key_pair)
        .context("could not self-sign TLS certificate")?;

    fs::write(&paths.cert_file, cert.pem())
        .with_context(|| format!("could not write {}", paths.cert_file.display()))?;
    fs::write(&paths.key_file, key_pair.serialize_pem())
        .with_context(|| format!("could not write {}", paths.key_file.display()))?;

    if !paths.cert_file.is_file() || !paths.key_file.is_file() {
        bail!("TLS files were not written");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_pem_pair_once() {
        let dir = tempfile::tempdir().unwrap();
        let first = ensure_self_signed(dir.path(), &[]).unwrap();
        assert!(first.cert_file.is_file());
        assert!(first.key_file.is_file());
        let cert = fs::read_to_string(&first.cert_file).unwrap();
        let key = fs::read_to_string(&first.key_file).unwrap();
        assert!(cert.contains("BEGIN CERTIFICATE"));
        assert!(key.contains("BEGIN") && key.contains("PRIVATE KEY"));

        let second = ensure_self_signed(dir.path(), &[IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1))])
            .unwrap();
        assert_eq!(first, second);
        // Existing pair is reused (no regenerate on extra IPs).
        assert_eq!(fs::read_to_string(&second.cert_file).unwrap(), cert);
    }
}
