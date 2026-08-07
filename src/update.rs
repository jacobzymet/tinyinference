//! Check GitHub Releases for a newer tinyinference build.

use std::{
    cmp::Ordering,
    sync::Mutex,
    time::{Duration, Instant},
};

use serde::Deserialize;

const RELEASES_API: &str =
    "https://api.github.com/repos/jacobzymet/tinyinference/releases?per_page=10";
const CHECK_TTL: Duration = Duration::from_secs(6 * 60 * 60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(4);

#[derive(Debug, Clone, serde::Serialize)]
pub struct UpdateNotice {
    /// Tag name as published on GitHub (`v0.5.0`).
    pub tag: String,
    /// Current running version (`0.4.0-beta`).
    pub current: String,
    pub name: String,
    pub html_url: String,
    pub prerelease: bool,
}

#[derive(Debug, Default)]
pub struct UpdateCache {
    inner: Mutex<CacheState>,
}

#[derive(Debug, Default)]
struct CacheState {
    checked_at: Option<Instant>,
    /// Last successful probe result (including “no update”).
    notice: Option<UpdateNotice>,
    /// In-flight / recently failed — avoid stacking requests.
    probing: bool,
}

impl UpdateCache {
    /// Cached notice only — never blocks on the network.
    pub fn peek(&self) -> Option<UpdateNotice> {
        let Ok(guard) = self.inner.lock() else {
            return None;
        };
        guard.notice.clone()
    }

    pub fn needs_refresh(&self) -> bool {
        let Ok(guard) = self.inner.lock() else {
            return false;
        };
        if guard.probing {
            return false;
        }
        match guard.checked_at {
            None => true,
            Some(at) => at.elapsed() >= CHECK_TTL,
        }
    }

    pub fn begin_probe(&self) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        if guard.probing {
            return false;
        }
        if let Some(at) = guard.checked_at
            && at.elapsed() < CHECK_TTL
        {
            return false;
        }
        guard.probing = true;
        true
    }

    pub fn finish_probe(&self, notice: Option<UpdateNotice>) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.checked_at = Some(Instant::now());
            guard.notice = notice;
            guard.probing = false;
        }
    }

    pub fn clear_probe_flag(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.probing = false;
        }
    }
}

/// Fetch the newest published GitHub release newer than `current` (e.g. `0.4.0-beta`).
pub fn check_for_update(current: &str) -> Option<UpdateNotice> {
    let current_ver = parse_version(current)?;
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(REQUEST_TIMEOUT))
        .user_agent(concat!("tinyinference/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent();

    let response = agent
        .get(RELEASES_API)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .call()
        .ok()?;
    if !(200..300).contains(&response.status().as_u16()) {
        return None;
    }
    let releases: Vec<GhRelease> = response.into_body().read_json().ok()?;

    let mut best: Option<(Version, GhRelease)> = None;
    for release in releases {
        if release.draft {
            continue;
        }
        let Some(tag_ver) = parse_version(&release.tag_name) else {
            continue;
        };
        if tag_ver <= current_ver {
            continue;
        }
        match &best {
            None => best = Some((tag_ver, release)),
            Some((prev, _)) if tag_ver > *prev => best = Some((tag_ver, release)),
            _ => {}
        }
    }

    best.map(|(_, release)| {
        let tag = release.tag_name;
        let name = if release.name.trim().is_empty() {
            tag.clone()
        } else {
            release.name
        };
        UpdateNotice {
            tag,
            current: current.trim().trim_start_matches('v').to_string(),
            name,
            html_url: release.html_url,
            prerelease: release.prerelease,
        }
    })
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    name: String,
    html_url: String,
    draft: bool,
    prerelease: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    /// `None` = final release; `Some` = prerelease label (`beta`, `beta.1`, …).
    pre: Option<String>,
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch)) {
            Ordering::Equal => match (&self.pre, &other.pre) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            },
            other => other,
        }
    }
}

fn parse_version(raw: &str) -> Option<Version> {
    let s = raw.trim().trim_start_matches('v');
    if s.is_empty() {
        return None;
    }
    let (core, pre) = match s.split_once('-') {
        Some((core, pre)) => (core, Some(pre.to_ascii_lowercase())),
        None => (s, None),
    };
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some(Version {
        major,
        minor,
        patch,
        pre,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_ordering() {
        assert!(parse_version("v0.5.0").unwrap() > parse_version("0.4.0-beta").unwrap());
        assert!(parse_version("0.4.0").unwrap() > parse_version("0.4.0-beta").unwrap());
        assert!(parse_version("0.4.1-beta").unwrap() > parse_version("0.4.0").unwrap());
        assert_eq!(
            parse_version("v1.2.3").unwrap(),
            parse_version("1.2.3").unwrap()
        );
    }
}
