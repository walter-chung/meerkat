//! `rkat run` with a stdin that is a pipe held open and silent, as some
//! launchers, supervisors and agent harnesses pass. Before the fix the default
//! `--stdin auto` read it to end of file before anything else and the run hung
//! with no output; each row here holds the pipe open for longer than its
//! deadline.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::panic
)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const DEADLINE: Duration = Duration::from_secs(30);

fn rkat_binary() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_rkat") {
        return path.into();
    }
    let mut path = std::env::current_exe().expect("current test executable");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push(if cfg!(windows) { "rkat.exe" } else { "rkat" });
    path
}

struct Roots {
    _temp: tempfile::TempDir,
    home: PathBuf,
    project: PathBuf,
    state: PathBuf,
    user: PathBuf,
}

fn roots() -> Roots {
    let temp = tempfile::TempDir::new().unwrap();
    let home = temp.path().to_path_buf();
    let project = temp.path().join("project");
    let state = temp.path().join("state");
    let user = temp.path().join("user");
    for dir in [&project, &state, &user, &project.join(".rkat")] {
        std::fs::create_dir_all(dir).unwrap();
    }
    Roots {
        _temp: temp,
        home,
        project,
        state,
        user,
    }
}

/// A stdio MCP server that never answers the handshake.
fn write_silent_mcp_server(project: &Path) {
    let stub = project.join("silent-mcp-server.sh");
    std::fs::write(&stub, "#!/bin/sh\nexec 2>/dev/null\nsleep 600\n").unwrap();
    std::fs::write(
        project.join(".rkat/mcp.toml"),
        format!(
            "[[servers]]\nname = \"silent\"\ncommand = \"/bin/sh\"\nargs = [\"{}\"]\nconnect_timeout_secs = 2\n",
            stub.display()
        ),
    )
    .unwrap();
}

/// Run `rkat` with `extra` root flags and `run_args`, its stdin a pipe this
/// test keeps open (never written, never closed) until the run ends or the
/// deadline passes. Returns the output, or panics at the deadline.
fn run_with_held_open_stdin(roots: &Roots, extra: &[&str], run_args: &[&str]) -> Output {
    let mut command = Command::new(rkat_binary());
    command
        .current_dir(&roots.project)
        .env("HOME", &roots.home)
        .env("RKAT_TEST_CLIENT", "1")
        .env_remove("RUST_LOG")
        .arg("--realm")
        .arg("stdin-held-open")
        .arg("--context-root")
        .arg(&roots.project)
        .arg("--state-root")
        .arg(&roots.state)
        .arg("--user-config-root")
        .arg(&roots.user)
        .args(extra)
        .arg("run")
        .args(run_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child: Child = command.spawn().expect("spawn rkat");
    // Hold the write end: the pipe stays open and silent for the whole row.
    let held_stdin = child.stdin.take().expect("piped stdin");
    // Drain both streams on their own threads, so a full pipe never stalls
    // rkat and a descendant still holding a stream never stalls the test.
    let stdout = drain(child.stdout.take().expect("piped stdout"));
    let stderr = drain(child.stderr.take().expect("piped stderr"));
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll rkat") {
            break status;
        }
        if started.elapsed() > DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "rkat run did not finish within {DEADLINE:?} with stdin held open; stdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&collect(stdout)),
                String::from_utf8_lossy(&collect(stderr))
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    drop(held_stdin);
    Output {
        status,
        stdout: collect(stdout),
        stderr: collect(stderr),
    }
}

fn drain(mut stream: impl Read + Send + 'static) -> mpsc::Receiver<Vec<u8>> {
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stream.read_to_end(&mut bytes);
        let _ = send.send(bytes);
    });
    receive
}

/// What a drained stream produced, waiting a few seconds at most for its end.
fn collect(stream: mpsc::Receiver<Vec<u8>>) -> Vec<u8> {
    stream
        .recv_timeout(Duration::from_secs(5))
        .unwrap_or_else(|_| b"<stream still held open by a descendant>".to_vec())
}

/// R1: the default `--stdin auto` gives up on a silent stdin after its grace
/// period, says so once, and runs the prompt.
#[test]
fn r1_auto_continues_without_a_held_open_silent_stdin() {
    let roots = roots();
    let output = run_with_held_open_stdin(&roots, &[], &["Reply with OK."]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "status={} stderr:\n{stderr}",
        output.status
    );
    assert_eq!(
        stderr.matches("no input arrived within").count(),
        1,
        "one warning naming the flags; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("--stdin blob") && stderr.contains("--stdin off"),
        "stderr:\n{stderr}"
    );
}

/// R2 (control): `--stdin off` never reads stdin.
#[test]
fn r2_stdin_off_ignores_a_held_open_stdin() {
    let roots = roots();
    let output = run_with_held_open_stdin(&roots, &[], &["--stdin", "off", "Reply with OK."]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "status={} stderr:\n{stderr}",
        output.status
    );
    assert!(
        !stderr.contains("no input arrived within"),
        "stderr:\n{stderr}"
    );
}

/// R3: the same under workspace confinement beside a stdio MCP server that
/// never answers: the run still finishes and the confinement report prints.
#[test]
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn r3_auto_under_workspace_confinement_with_a_silent_stdio_server() {
    let roots = roots();
    write_silent_mcp_server(&roots.project);
    let output =
        run_with_held_open_stdin(&roots, &["--confinement", "workspace"], &["Reply with OK."]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "status={} stderr:\n{stderr}",
        output.status
    );
    assert!(
        stderr.contains("mcp_stdio confinement: requested workspace"),
        "the confinement report prints; stderr:\n{stderr}"
    );
    assert_eq!(
        stderr.matches("no input arrived within").count(),
        1,
        "stderr:\n{stderr}"
    );
}
