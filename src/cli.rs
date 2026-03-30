use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "voxtral",
    version,
    about = "Low-latency local Voxtral TTS CLI for Apple Silicon"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Download,
    Serve {
        #[arg(long)]
        foreground: bool,
    },
    Speak {
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long, value_enum, default_value_t = SynthesisProfile::Balanced)]
        profile: SynthesisProfile,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, help = "Generate audio without playing it locally")]
        no_play: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Wav)]
        format: OutputFormat,
    },
    Voices,
    Status,
    Stop,
    Doctor,
    Bench {
        #[arg(long, value_enum, default_value_t = BenchScenarioArg::All)]
        scenario: BenchScenarioArg,
        #[arg(long, value_enum, default_value_t = SynthesisProfile::Balanced)]
        profile: SynthesisProfile,
        #[arg(long, default_value_t = 5)]
        runs: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    Wav,
    Flac,
    Mp3,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BenchScenarioArg {
    Short,
    Medium,
    Long,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SynthesisProfile {
    Quality,
    Balanced,
    Fast,
}

impl SynthesisProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Fast => "fast",
        }
    }
}
