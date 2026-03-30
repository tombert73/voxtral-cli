#[tokio::main]
async fn main() {
    if let Err(err) = voxtral_cli::app::run().await {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
