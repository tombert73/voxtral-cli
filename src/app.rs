use crate::{
    bench,
    cli::{Cli, Command, OutputFormat, SynthesisProfile},
    config::{AppPaths, Config, ensure_parent},
    protocol::{Request, Response},
};
use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use std::{
    fs,
    io::{self, IsTerminal, Read},
    os::unix::fs::PermissionsExt,
    os::unix::process::CommandExt,
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{Child, Command as ProcessCommand, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    time::sleep,
};

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    let paths = AppPaths::detect()?;
    let config = Config::load_or_create(&paths)?;

    match cli.command {
        Command::Download => download(&paths, &config).await,
        Command::Serve { foreground } => serve(&paths, &config, foreground).await,
        Command::Speak {
            text,
            voice,
            profile,
            output,
            no_play,
            format,
        } => {
            speak(
                &paths, &config, text, voice, profile, output, no_play, format,
            )
            .await
        }
        Command::Voices => voices(&config).await,
        Command::Status => status(&config).await,
        Command::Stop => stop(&config).await,
        Command::Doctor => doctor(&paths, &config).await,
        Command::Bench {
            scenario,
            profile,
            runs,
        } => bench(&paths, &config, scenario, profile, runs).await,
    }
}

async fn download(paths: &AppPaths, config: &Config) -> Result<()> {
    paths.ensure_dirs()?;
    ensure_parent(&config.python_venv)?;
    if !config.python_bin().exists() {
        run_command(
            ProcessCommand::new("uv")
                .arg("venv")
                .arg(&config.python_venv)
                .arg("--python")
                .arg("python3"),
            "creating Python virtual environment",
        )?;
    }
    run_command(
        ProcessCommand::new("uv")
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(config.python_bin())
            .arg("-r")
            .arg(Config::requirements_file(paths)),
        "installing Python runtime dependencies",
    )?;
    run_helper_command(
        &config.python_bin(),
        &Config::helper_script(paths),
        &[
            "download",
            "--model-repo",
            &config.model_repo,
            "--model-dir",
            config.model_dir.to_string_lossy().as_ref(),
        ],
        "downloading quantized model weights",
    )?;
    println!("download complete");
    println!("model: {}", config.model_repo);
    println!("path: {}", config.model_dir.display());
    Ok(())
}

async fn serve(paths: &AppPaths, config: &Config, foreground: bool) -> Result<()> {
    paths.ensure_dirs()?;
    if let Ok(response) = send_request(config, &Request::Ping).await {
        if response.ok {
            println!("daemon already running at {}", config.socket_path.display());
            return Ok(());
        }
    }
    if !config.python_bin().exists() {
        bail!("runtime missing. run `voxtral download` first");
    }
    if !config.model_dir.exists() {
        bail!(
            "model missing at {}. run `voxtral download` first",
            config.model_dir.display()
        );
    }
    cleanup_stale_runtime_files(config)?;
    ensure_parent(&config.socket_path)?;
    ensure_parent(&config.pid_file)?;
    ensure_parent(&config.log_file)?;
    let helper = Config::helper_script(paths);
    let mut command = daemon_command(config, &helper);

    if foreground {
        let status = command
            .status()
            .context("failed to launch foreground daemon")?;
        if !status.success() {
            bail!("daemon exited with status {}", render_status(status));
        }
        return Ok(());
    }

    let mut child = spawn_background_daemon(config, &helper)?;
    wait_for_daemon_ready(config, &mut child, Duration::from_secs(45)).await?;
    println!("daemon ready");
    println!("socket: {}", config.socket_path.display());
    Ok(())
}

async fn speak(
    paths: &AppPaths,
    config: &Config,
    text_arg: Option<String>,
    voice_arg: Option<String>,
    profile: SynthesisProfile,
    output: Option<PathBuf>,
    no_play: bool,
    format: OutputFormat,
) -> Result<()> {
    let text = resolve_text(text_arg)?;
    ensure_daemon(paths, config).await?;
    let voice = voice_arg.unwrap_or_else(|| config.default_voice.clone());
    let play = resolve_playback(config, no_play);
    let output_path = output.unwrap_or_else(|| default_output_path(paths, format.extension()));
    ensure_parent(&output_path)?;
    let response = send_request(
        config,
        &Request::Speak {
            text,
            voice,
            profile: profile.as_str().to_string(),
            output_path: output_path.clone(),
            format: format.extension().to_string(),
        },
    )
    .await?;
    require_ok(&response)?;
    let actual_output = response
        .output_path
        .clone()
        .ok_or_else(|| anyhow!("daemon returned no output path"))?;
    println!(
        "generated {} in {:.2} ms ({:.2}x real time)",
        actual_output.display(),
        response.wall_ms.unwrap_or_default(),
        speed_from_response(&response)
    );
    if play {
        run_command(
            ProcessCommand::new("afplay").arg(&actual_output),
            "playing generated audio",
        )
        .with_context(|| format!("saved audio at {}", actual_output.display()))?;
        println!("played {}", actual_output.display());
    }
    Ok(())
}

async fn voices(config: &Config) -> Result<()> {
    let response = send_request(config, &Request::ListVoices).await?;
    require_ok(&response)?;
    for voice in response.voices.unwrap_or_default() {
        println!("{voice}");
    }
    Ok(())
}

async fn status(config: &Config) -> Result<()> {
    match send_request(config, &Request::Ping).await {
        Ok(response) if response.ok => {
            println!("running");
            println!("backend: {}", response.backend.unwrap_or_default());
            println!("model: {}", response.model.unwrap_or_default());
            println!("socket: {}", config.socket_path.display());
            Ok(())
        }
        Ok(response) => {
            println!("not healthy");
            if let Some(error) = response.error {
                println!("error: {error}");
            }
            Ok(())
        }
        Err(_) => {
            println!("not running");
            Ok(())
        }
    }
}

async fn stop(config: &Config) -> Result<()> {
    if let Ok(response) = send_request(config, &Request::Shutdown).await {
        require_ok(&response)?;
        wait_for_daemon_stop(config, Duration::from_secs(5)).await?;
        println!("shutdown requested");
    } else if let Some(pid) = read_pid(&config.pid_file)? {
        if !pid_matches_helper(pid)? {
            bail!(
                "refusing to signal pid {} because it does not look like the Voxtral helper",
                pid
            );
        }
        kill(Pid::from_raw(pid as i32), Signal::SIGTERM)
            .with_context(|| format!("failed to stop pid {pid}"))?;
        wait_for_daemon_stop(config, Duration::from_secs(5)).await?;
        println!("sent SIGTERM to {pid}");
    } else {
        println!("daemon not running");
    }
    Ok(())
}

async fn doctor(paths: &AppPaths, config: &Config) -> Result<()> {
    let state = inspect_runtime_state(config).await?;
    println!("config: {}", paths.config_file.display());
    println!("config dir mode: {}", format_mode(&paths.config_dir)?);
    println!("python: {}", config.python_bin().display());
    println!("model dir: {}", config.model_dir.display());
    println!("socket: {}", config.socket_path.display());
    println!("log: {}", config.log_file.display());
    println!("python runtime present: {}", state.python_runtime_present);
    println!("model present: {}", state.model_present);
    println!("daemon socket present: {}", state.socket_present);
    println!("daemon reachable: {}", state.daemon_reachable);
    println!("pid file present: {}", state.pid_present);
    println!("pid matches helper: {}", state.pid_matches_helper);
    println!(
        "runtime dir mode: {}",
        format_mode(config.pid_file.parent().unwrap_or(&paths.data_dir))?
    );
    println!("log file present: {}", state.log_present);
    println!("default profile: {}", config.default_profile);
    println!("default playback: {}", config.default_playback);
    if state.stale_runtime_state {
        println!("warning: stale runtime state detected");
    }
    let excerpt = tail_log_excerpt(&config.log_file, 10)?;
    if !excerpt.trim().is_empty() {
        println!("recent log excerpt:");
        println!("{excerpt}");
    }
    Ok(())
}

async fn bench(
    paths: &AppPaths,
    config: &Config,
    scenario_arg: crate::cli::BenchScenarioArg,
    profile: SynthesisProfile,
    runs: usize,
) -> Result<()> {
    ensure_daemon(paths, config).await?;
    let scenarios = bench::load_scenarios(scenario_arg);
    if scenarios.is_empty() {
        bail!("no benchmark scenarios selected");
    }
    let mut summaries = Vec::new();
    for scenario in scenarios {
        let warmup_output = default_output_path(paths, "wav");
        let warmup = send_request(
            config,
            &Request::Speak {
                text: "Warm up.".to_string(),
                voice: config.default_voice.clone(),
                profile: profile.as_str().to_string(),
                output_path: warmup_output.clone(),
                format: "wav".to_string(),
            },
        )
        .await?;
        require_ok(&warmup)?;
        let _ = fs::remove_file(warmup_output);

        let mut run_rows = Vec::new();
        for idx in 0..runs {
            let output = default_output_path(paths, "wav");
            let response = send_request(
                config,
                &Request::Speak {
                    text: scenario.text.clone(),
                    voice: config.default_voice.clone(),
                    profile: profile.as_str().to_string(),
                    output_path: output.clone(),
                    format: "wav".to_string(),
                },
            )
            .await?;
            require_ok(&response)?;
            let audio_seconds = response
                .audio_seconds
                .ok_or_else(|| anyhow!("missing audio duration"))?;
            let wall_seconds = response
                .wall_ms
                .ok_or_else(|| anyhow!("missing wall time"))?
                / 1000.0;
            run_rows.push(bench::build_run(idx + 1, audio_seconds, wall_seconds));
            let _ = fs::remove_file(output);
        }
        summaries.push(bench::summarize(&scenario.name, &scenario.text, run_rows));
    }

    if let Some(parent) = config.benchmark_json.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let report = bench::BenchmarkReport {
        model: config.model_repo.clone(),
        backend: config.backend.clone(),
        profile: profile.as_str().to_string(),
        scenarios: summaries,
    };
    fs::write(&config.benchmark_json, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("failed to write {}", config.benchmark_json.display()))?;

    for summary in &report.scenarios {
        println!(
            "{}: chars={} best={:.2}x median={:.2}x best_rtf={:.3} median_rtf={:.3}",
            summary.name,
            summary.char_count,
            summary.best_speed_x,
            summary.median_speed_x,
            summary.best_rtf,
            summary.median_rtf
        );
    }
    println!("json: {}", config.benchmark_json.display());
    Ok(())
}

async fn ensure_daemon(paths: &AppPaths, config: &Config) -> Result<()> {
    if send_request(config, &Request::Ping).await.is_ok() {
        return Ok(());
    }
    if config.auto_start {
        serve(paths, config, false).await?;
        Ok(())
    } else {
        bail!("daemon not running. start it with `voxtral serve`");
    }
}

async fn send_request(config: &Config, request: &Request) -> Result<Response> {
    let stream = tokio::time::timeout(
        Duration::from_secs(3),
        UnixStream::connect(&config.socket_path),
    )
    .await
    .context("timed out connecting to daemon")?
    .with_context(|| format!("failed to connect to {}", config.socket_path.display()))?;
    let (read_half, mut write_half) = stream.into_split();
    let payload = serde_json::to_vec(request).context("failed to encode request")?;
    tokio::time::timeout(Duration::from_secs(3), write_half.write_all(&payload))
        .await
        .context("timed out writing request")?
        .context("write failed")?;
    tokio::time::timeout(Duration::from_secs(3), write_half.write_all(b"\n"))
        .await
        .context("timed out writing request terminator")?
        .context("write failed")?;
    tokio::time::timeout(Duration::from_secs(3), write_half.flush())
        .await
        .context("timed out flushing request")?
        .context("flush failed")?;
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(30), reader.read_line(&mut line))
        .await
        .context("timed out waiting for daemon response")?
        .context("read failed")?;
    if line.trim().is_empty() {
        bail!("daemon returned an empty response");
    }
    let response = serde_json::from_str::<Response>(&line).context("invalid daemon response")?;
    Ok(response)
}

fn run_command(command: &mut ProcessCommand, action: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed while {action}"))?;
    if status.success() {
        return Ok(());
    }
    bail!("{action} failed with status {}", render_status(status));
}

fn run_helper_command(python: &Path, script: &Path, args: &[&str], action: &str) -> Result<()> {
    let mut command = ProcessCommand::new(python);
    command.arg(script);
    for arg in args {
        command.arg(arg);
    }
    run_command(&mut command, action)
}

fn render_status(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => code.to_string(),
        None => match status.signal() {
            Some(signal) => format!("signal {signal}"),
            None => "unknown".to_string(),
        },
    }
}

async fn wait_for_daemon_ready(
    config: &Config,
    child: &mut Child,
    timeout: Duration,
) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        if let Ok(response) = send_request(config, &Request::Ping).await {
            if response.ok {
                return Ok(());
            }
        }

        if let Some(status) = child
            .try_wait()
            .context("failed to inspect daemon child process")?
        {
            let excerpt = tail_log_excerpt(&config.log_file, 40)?;
            bail!(
                "daemon exited before becoming ready with status {}{}",
                render_status(status),
                format_log_excerpt(&excerpt)
            );
        }

        if start.elapsed() > timeout {
            let excerpt = tail_log_excerpt(&config.log_file, 40)?;
            bail!(
                "timed out waiting for daemon at {}{}",
                config.socket_path.display(),
                format_log_excerpt(&excerpt)
            );
        }

        sleep(Duration::from_millis(250)).await;
    }
}

async fn wait_for_daemon_stop(config: &Config, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        let reachable = send_request(config, &Request::Ping).await.is_ok();
        let pid_exists = config.pid_file.exists();
        let socket_exists = config.socket_path.exists();
        if !reachable && !pid_exists && !socket_exists {
            return Ok(());
        }
        if start.elapsed() > timeout {
            let excerpt = tail_log_excerpt(&config.log_file, 20)?;
            bail!(
                "daemon did not stop cleanly{}",
                format_log_excerpt(&excerpt)
            );
        }
        sleep(Duration::from_millis(100)).await;
    }
}

fn resolve_text(text_arg: Option<String>) -> Result<String> {
    if let Some(text) = text_arg {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            bail!("--text cannot be empty");
        }
        return Ok(trimmed.to_string());
    }
    let stdin = io::stdin();
    if stdin.is_terminal() {
        bail!("no input text provided. pass --text or pipe text into `voxtral speak`");
    }
    let mut buffer = String::new();
    stdin
        .lock()
        .read_to_string(&mut buffer)
        .context("failed to read stdin")?;
    let trimmed = buffer.trim();
    if trimmed.is_empty() {
        bail!("piped stdin was empty");
    }
    Ok(trimmed.to_string())
}

fn default_output_path(paths: &AppPaths, extension: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    paths
        .output_dir
        .join(format!("voxtral-{millis}.{extension}"))
}

fn speed_from_response(response: &Response) -> f64 {
    match (response.audio_seconds, response.wall_ms) {
        (Some(audio), Some(wall_ms)) if wall_ms > 0.0 => audio / (wall_ms / 1000.0),
        _ => 0.0,
    }
}

fn resolve_playback(config: &Config, no_play: bool) -> bool {
    config.default_playback && !no_play
}

fn require_ok(response: &Response) -> Result<()> {
    if response.ok {
        Ok(())
    } else {
        Err(anyhow!(
            "{}",
            response
                .error
                .clone()
                .unwrap_or_else(|| "unknown daemon error".to_string())
        ))
    }
}

fn read_pid(path: &Path) -> Result<Option<u32>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let pid = raw.trim().parse::<u32>().context("invalid pid file")?;
    Ok(Some(pid))
}

fn cleanup_stale_runtime_files(config: &Config) -> Result<()> {
    match read_pid(&config.pid_file) {
        Ok(Some(pid)) => {
            if pid_matches_helper(pid)? {
                let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
                std::thread::sleep(Duration::from_millis(250));
            }
            let _ = fs::remove_file(&config.pid_file);
            let _ = fs::remove_file(&config.socket_path);
        }
        Ok(None) => {
            if config.socket_path.exists() {
                let _ = fs::remove_file(&config.socket_path);
            }
        }
        Err(_) => {
            let _ = fs::remove_file(&config.pid_file);
            let _ = fs::remove_file(&config.socket_path);
        }
    }
    Ok(())
}

struct RuntimeState {
    python_runtime_present: bool,
    model_present: bool,
    socket_present: bool,
    daemon_reachable: bool,
    pid_present: bool,
    pid_matches_helper: bool,
    log_present: bool,
    stale_runtime_state: bool,
}

async fn inspect_runtime_state(config: &Config) -> Result<RuntimeState> {
    let python_runtime_present = config.python_bin().exists();
    let model_present = config.model_dir.exists();
    let socket_present = config.socket_path.exists();
    let daemon_reachable = if socket_present {
        send_request(config, &Request::Ping).await.is_ok()
    } else {
        false
    };
    let pid = read_pid(&config.pid_file).ok().flatten();
    let pid_present = pid.is_some();
    let pid_matches_helper = match pid {
        Some(pid) => pid_matches_helper(pid)?,
        None => false,
    };
    let log_present = config.log_file.exists();
    let stale_runtime_state = (socket_present || pid_present) && !daemon_reachable;

    Ok(RuntimeState {
        python_runtime_present,
        model_present,
        socket_present,
        daemon_reachable,
        pid_present,
        pid_matches_helper,
        log_present,
        stale_runtime_state,
    })
}

fn pid_matches_helper(pid: u32) -> Result<bool> {
    let output = ProcessCommand::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
        .context("failed to inspect pid")?;
    if !output.status.success() {
        return Ok(false);
    }
    let command = String::from_utf8_lossy(&output.stdout);
    Ok(command.contains("voxtral_helper.py"))
}

fn daemon_command(config: &Config, helper: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new(config.python_bin());
    command
        .arg(helper)
        .arg("daemon")
        .arg("--socket")
        .arg(&config.socket_path)
        .arg("--pid-file")
        .arg(&config.pid_file)
        .arg("--model-repo")
        .arg(&config.model_repo)
        .arg("--model-dir")
        .arg(&config.model_dir);
    command
}

fn spawn_background_daemon(config: &Config, helper: &Path) -> Result<Child> {
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.log_file)
        .with_context(|| format!("failed to open {}", config.log_file.display()))?;
    fs::set_permissions(&config.log_file, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("failed to set permissions on {}", config.log_file.display()))?;
    let err_log = log
        .try_clone()
        .context("failed to clone daemon log handle")?;

    let mut command = daemon_command(config, helper);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err_log));
    unsafe {
        command.pre_exec(|| {
            nix::unistd::setsid().map_err(|err| io::Error::other(err.to_string()))?;
            Ok(())
        });
    }
    command.spawn().context("failed to spawn background daemon")
}

fn tail_log_excerpt(path: &Path, lines: usize) -> Result<String> {
    if !path.exists() {
        return Ok(String::new());
    }
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let excerpt = raw
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    Ok(excerpt)
}

fn format_log_excerpt(excerpt: &str) -> String {
    if excerpt.trim().is_empty() {
        String::new()
    } else {
        format!("\n\nRecent daemon log output:\n{excerpt}")
    }
}

fn format_mode(path: &Path) -> Result<String> {
    if !path.exists() {
        return Ok("missing".to_string());
    }
    let mode = fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .permissions()
        .mode()
        & 0o777;
    Ok(format!("{mode:#05o}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, tempdir};

    fn test_paths() -> AppPaths {
        AppPaths {
            config_file: PathBuf::from("/tmp/config.json"),
            config_dir: PathBuf::from("/tmp"),
            data_dir: PathBuf::from("/tmp"),
            cache_dir: PathBuf::from("/tmp"),
            output_dir: PathBuf::from("/tmp"),
            model_root: PathBuf::from("/tmp"),
            python_dir: PathBuf::from("/tmp"),
        }
    }

    #[test]
    fn text_argument_wins() {
        let text = resolve_text(Some(" hello ".to_string())).expect("text should resolve");
        assert_eq!(text, "hello");
    }

    #[test]
    fn empty_text_argument_fails() {
        let err = resolve_text(Some("   ".to_string())).expect_err("empty text should fail");
        assert!(err.to_string().contains("--text cannot be empty"));
    }

    #[test]
    fn speed_calculation_matches_expected_ratio() {
        let response = Response {
            ok: true,
            error: None,
            backend: None,
            model: None,
            output_path: None,
            voices: None,
            audio_seconds: Some(12.0),
            wall_ms: Some(1200.0),
            queue_ms: Some(0.0),
        };
        assert!((speed_from_response(&response) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn playback_defaults_to_config_value() {
        let config = Config::default_for(&test_paths());
        assert!(resolve_playback(&config, false));
        assert!(!resolve_playback(&config, true));
    }

    #[test]
    fn synthesis_profile_string_values_are_stable() {
        assert_eq!(SynthesisProfile::Quality.as_str(), "quality");
        assert_eq!(SynthesisProfile::Balanced.as_str(), "balanced");
        assert_eq!(SynthesisProfile::Fast.as_str(), "fast");
    }

    #[test]
    fn default_config_uses_balanced_profile() {
        let config = Config::default_for(&test_paths());
        assert_eq!(config.default_profile, "balanced");
    }

    #[test]
    fn require_ok_returns_response_error_message() {
        let response = Response {
            ok: false,
            error: Some("daemon crashed during startup".to_string()),
            backend: None,
            model: None,
            output_path: None,
            voices: None,
            audio_seconds: None,
            wall_ms: None,
            queue_ms: None,
        };
        let err = require_ok(&response).expect_err("error response should fail");
        assert!(err.to_string().contains("daemon crashed during startup"));
    }

    #[test]
    fn read_pid_returns_none_for_missing_file() {
        let temp = NamedTempFile::new().expect("temp file");
        let path = temp.path().with_extension("missing");
        let pid = read_pid(&path).expect("missing pid file should be fine");
        assert!(pid.is_none());
    }

    #[test]
    fn cleanup_stale_runtime_files_removes_invalid_pid_and_socket() {
        let dir = tempdir().expect("temp dir");
        let runtime = dir.path().join("runtime");
        fs::create_dir_all(&runtime).expect("runtime dir");
        let config = Config {
            backend: "mlx_daemon".to_string(),
            model_repo: "test-model".to_string(),
            model_dir: dir.path().join("model"),
            python_venv: dir.path().join("venv"),
            socket_path: runtime.join("voxtral.sock"),
            pid_file: runtime.join("voxtral.pid"),
            log_file: runtime.join("voxtral.log"),
            default_voice: "casual_female".to_string(),
            default_format: "wav".to_string(),
            default_profile: "balanced".to_string(),
            default_playback: true,
            auto_start: true,
            benchmark_json: dir.path().join("bench.json"),
        };
        fs::write(&config.pid_file, "not-a-pid").expect("write pid");
        fs::write(&config.socket_path, "").expect("write socket placeholder");

        cleanup_stale_runtime_files(&config).expect("cleanup should succeed");

        assert!(!config.pid_file.exists());
        assert!(!config.socket_path.exists());
    }

    #[test]
    fn format_log_excerpt_is_empty_for_blank_input() {
        assert!(format_log_excerpt("").is_empty());
        assert!(format_log_excerpt("   ").is_empty());
    }

    #[test]
    fn tail_log_excerpt_returns_recent_lines_only() {
        let file = NamedTempFile::new().expect("temp log");
        fs::write(file.path(), "line1\nline2\nline3\n").expect("write log");
        let excerpt = tail_log_excerpt(file.path(), 2).expect("read excerpt");
        assert_eq!(excerpt, "line2\nline3");
    }

    #[test]
    fn format_mode_returns_missing_for_absent_path() {
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("missing");
        let mode = format_mode(&path).expect("format mode");
        assert_eq!(mode, "missing");
    }

    #[tokio::test]
    async fn inspect_runtime_state_detects_stale_pid() {
        let dir = tempdir().expect("temp dir");
        let runtime = dir.path().join("runtime");
        fs::create_dir_all(&runtime).expect("runtime dir");
        let config = Config {
            backend: "mlx_daemon".to_string(),
            model_repo: "test-model".to_string(),
            model_dir: dir.path().join("model"),
            python_venv: dir.path().join("venv"),
            socket_path: runtime.join("voxtral.sock"),
            pid_file: runtime.join("voxtral.pid"),
            log_file: runtime.join("voxtral.log"),
            default_voice: "casual_female".to_string(),
            default_format: "wav".to_string(),
            default_profile: "balanced".to_string(),
            default_playback: true,
            auto_start: true,
            benchmark_json: dir.path().join("bench.json"),
        };
        fs::write(&config.pid_file, "999999").expect("write pid");
        let state = inspect_runtime_state(&config).await.expect("inspect state");
        assert!(state.pid_present);
        assert!(!state.pid_matches_helper);
        assert!(state.stale_runtime_state);
    }
}
