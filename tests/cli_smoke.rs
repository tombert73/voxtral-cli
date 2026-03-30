use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_voxtral")
}

#[test]
fn root_help_lists_core_commands() {
    let output = Command::new(binary())
        .arg("--help")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("serve"));
    assert!(stdout.contains("speak"));
    assert!(stdout.contains("bench"));
}

#[test]
fn speak_help_documents_no_play() {
    let output = Command::new(binary())
        .args(["speak", "--help"])
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--no-play"));
    assert!(stdout.contains("--profile"));
    assert!(stdout.contains("--text"));
    assert!(stdout.contains("--output"));
}

#[test]
fn doctor_command_runs_successfully() {
    let output = Command::new(binary())
        .arg("doctor")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("config:"));
    assert!(stdout.contains("python runtime present:"));
    assert!(stdout.contains("default profile:"));
}

#[test]
fn status_command_runs_successfully() {
    let output = Command::new(binary())
        .arg("status")
        .output()
        .expect("binary should run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("running")
            || stdout.contains("not running")
            || stdout.contains("not healthy")
    );
}
