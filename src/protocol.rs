use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Request {
    Ping,
    Shutdown,
    ListVoices,
    Speak {
        text: String,
        voice: String,
        profile: String,
        output_path: PathBuf,
        format: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub error: Option<String>,
    pub backend: Option<String>,
    pub model: Option<String>,
    pub output_path: Option<PathBuf>,
    pub voices: Option<Vec<String>>,
    pub audio_seconds: Option<f64>,
    pub wall_ms: Option<f64>,
    pub queue_ms: Option<f64>,
}
