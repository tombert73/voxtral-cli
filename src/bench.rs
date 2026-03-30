use crate::cli::BenchScenarioArg;
use serde::{Deserialize, Serialize};

const PROMPTS_JSON: &str = include_str!("../benchmarks/prompts.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    pub name: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub run_index: usize,
    pub audio_seconds: f64,
    pub wall_seconds: f64,
    pub rtf: f64,
    pub speed_x: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub name: String,
    pub char_count: usize,
    pub best_speed_x: f64,
    pub median_speed_x: f64,
    pub best_rtf: f64,
    pub median_rtf: f64,
    pub runs: Vec<BenchmarkRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub model: String,
    pub backend: String,
    pub profile: String,
    pub scenarios: Vec<BenchmarkSummary>,
}

pub fn load_scenarios(arg: BenchScenarioArg) -> Vec<BenchmarkScenario> {
    let mut scenarios: Vec<BenchmarkScenario> =
        serde_json::from_str(PROMPTS_JSON).expect("invalid benchmark prompt JSON");
    scenarios.retain(|scenario| match arg {
        BenchScenarioArg::Short => scenario.name == "short",
        BenchScenarioArg::Medium => scenario.name == "medium",
        BenchScenarioArg::Long => scenario.name == "long",
        BenchScenarioArg::All => true,
    });
    scenarios
}

pub fn build_run(run_index: usize, audio_seconds: f64, wall_seconds: f64) -> BenchmarkRun {
    let rtf = wall_seconds / audio_seconds;
    let speed_x = audio_seconds / wall_seconds;
    BenchmarkRun {
        run_index,
        audio_seconds,
        wall_seconds,
        rtf,
        speed_x,
    }
}

pub fn summarize(name: &str, text: &str, runs: Vec<BenchmarkRun>) -> BenchmarkSummary {
    let mut speed_values: Vec<f64> = runs.iter().map(|run| run.speed_x).collect();
    let mut rtf_values: Vec<f64> = runs.iter().map(|run| run.rtf).collect();
    speed_values.sort_by(|a, b| a.total_cmp(b));
    rtf_values.sort_by(|a, b| a.total_cmp(b));
    let median_idx = speed_values.len() / 2;
    BenchmarkSummary {
        name: name.to_string(),
        char_count: text.chars().count(),
        best_speed_x: runs
            .iter()
            .map(|run| run.speed_x)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or_default(),
        median_speed_x: *speed_values.get(median_idx).unwrap_or(&0.0),
        best_rtf: runs
            .iter()
            .map(|run| run.rtf)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or_default(),
        median_rtf: *rtf_values.get(median_idx).unwrap_or(&0.0),
        runs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_filtering_works() {
        assert_eq!(load_scenarios(BenchScenarioArg::Short).len(), 1);
        assert_eq!(load_scenarios(BenchScenarioArg::All).len(), 3);
    }

    #[test]
    fn summary_uses_sorted_median() {
        let runs = vec![
            build_run(1, 10.0, 1.0),
            build_run(2, 10.0, 2.0),
            build_run(3, 10.0, 4.0),
        ];
        let summary = summarize("short", "hello", runs);
        assert!((summary.best_speed_x - 10.0).abs() < f64::EPSILON);
        assert!((summary.median_speed_x - 5.0).abs() < f64::EPSILON);
        assert!((summary.best_rtf - 0.1).abs() < f64::EPSILON);
        assert!((summary.median_rtf - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn summary_uses_upper_middle_for_even_run_lists() {
        let runs = vec![
            build_run(1, 10.0, 1.0),
            build_run(2, 10.0, 2.0),
            build_run(3, 10.0, 4.0),
            build_run(4, 10.0, 5.0),
        ];
        let summary = summarize("short", "hello", runs);
        assert!((summary.median_speed_x - 5.0).abs() < f64::EPSILON);
        assert!((summary.median_rtf - 0.4).abs() < f64::EPSILON);
    }
}
