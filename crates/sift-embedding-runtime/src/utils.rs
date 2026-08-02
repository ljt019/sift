use std::str::FromStr;

pub fn get_num_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| usize::from_str(&value).ok())
        .filter(|&count| count > 0)
        .unwrap_or_else(|| num_cpus::get_physical().max(1))
}
