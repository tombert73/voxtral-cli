use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

pub const DEFAULT_MODEL_REPO: &str = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit";
pub const DEFAULT_VOICE: &str = "casual_female";
pub const DEFAULT_PROFILE: &str = "balanced";

fn default_true() -> bool {
    true
}

fn default_profile_string() -> String {
    DEFAULT_PROFILE.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub backend: String,
    pub model_repo: String,
    pub model_dir: PathBuf,
    pub python_venv: PathBuf,
    pub socket_path: PathBuf,
    pub pid_file: PathBuf,
    pub log_file: PathBuf,
    pub default_voice: String,
    pub default_format: String,
    #[serde(default = "default_profile_string")]
    pub default_profile: String,
    #[serde(default = "default_true")]
    pub default_playback: bool,
    #[serde(default = "default_true")]
    pub auto_start: bool,
    pub benchmark_json: PathBuf,
}

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub config_file: PathBuf,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub model_root: PathBuf,
    pub python_dir: PathBuf,
}

impl AppPaths {
    pub fn detect() -> Result<Self> {
        let dirs = ProjectDirs::from("ai", "lucataco", "voxtral-cli")
            .context("failed to determine application directories")?;
        let config_dir = dirs.config_dir().to_path_buf();
        let data_dir = dirs.data_local_dir().to_path_buf();
        let cache_dir = dirs.cache_dir().to_path_buf();
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            config_file: config_dir.join("config.json"),
            config_dir,
            data_dir: data_dir.clone(),
            cache_dir: cache_dir.clone(),
            output_dir: data_dir.join("outputs"),
            model_root: cache_dir.join("models"),
            python_dir: manifest_dir.join("python"),
        })
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.cache_dir,
            &self.output_dir,
            &self.model_root,
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        restrict_permissions(&self.config_dir)?;
        restrict_permissions(&self.data_dir)?;
        Ok(())
    }
}

impl Config {
    pub fn load_or_create(paths: &AppPaths) -> Result<Self> {
        paths.ensure_dirs()?;
        if paths.config_file.exists() {
            let raw = fs::read_to_string(&paths.config_file)
                .with_context(|| format!("failed to read {}", paths.config_file.display()))?;
            let cfg = serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse {}", paths.config_file.display()))?;
            Ok(cfg)
        } else {
            let cfg = Self::default_for(paths);
            cfg.save(paths)?;
            Ok(cfg)
        }
    }

    pub fn default_for(paths: &AppPaths) -> Self {
        let model_dir = paths.model_root.join(sanitize_repo(DEFAULT_MODEL_REPO));
        let python_venv = paths.data_dir.join("runtime").join(".venv");
        let runtime_dir = paths.data_dir.join("runtime");
        Self {
            backend: "mlx_daemon".to_string(),
            model_repo: DEFAULT_MODEL_REPO.to_string(),
            model_dir,
            python_venv: python_venv.clone(),
            socket_path: runtime_dir.join("voxtral.sock"),
            pid_file: runtime_dir.join("voxtral.pid"),
            log_file: runtime_dir.join("voxtral.log"),
            default_voice: DEFAULT_VOICE.to_string(),
            default_format: "wav".to_string(),
            default_profile: DEFAULT_PROFILE.to_string(),
            default_playback: true,
            auto_start: true,
            benchmark_json: paths.data_dir.join("benchmarks").join("latest.json"),
        }
    }

    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        if let Some(parent) = paths.config_file.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let raw = serde_json::to_string_pretty(self).context("failed to serialize config")?;
        fs::write(&paths.config_file, raw)
            .with_context(|| format!("failed to write {}", paths.config_file.display()))?;
        Ok(())
    }

    pub fn python_bin(&self) -> PathBuf {
        self.python_venv.join("bin").join("python")
    }

    pub fn requirements_file(paths: &AppPaths) -> PathBuf {
        paths.python_dir.join("requirements.txt")
    }

    pub fn helper_script(paths: &AppPaths) -> PathBuf {
        paths.python_dir.join("voxtral_helper.py")
    }
}

pub fn sanitize_repo(repo: &str) -> String {
    repo.replace('/', "--")
}

pub fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        restrict_permissions(parent)?;
    }
    Ok(())
}

fn restrict_permissions(path: &Path) -> Result<()> {
    let perms = fs::Permissions::from_mode(0o700);
    fs::set_permissions(path, perms)
        .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    Ok(())
}
