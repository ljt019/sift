use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;

use super::types::{Dependency, HealthResponse, Status};
use crate::state::AppState;

/// Health checks must stay fast, never inherit the search timeout.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

pub async fn health(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let searxng = probe(&state).await;

    let (status, code) = if searxng.reachable {
        (Status::Ok, StatusCode::OK)
    } else {
        (Status::Degraded, StatusCode::SERVICE_UNAVAILABLE)
    };

    let body = HealthResponse {
        status,
        version: env!("CARGO_PKG_VERSION"),
        searxng,
    };

    (code, Json(body))
}

async fn probe(state: &AppState) -> Dependency {
    let url = state.config.searxng_url.clone();
    let started = Instant::now();

    let result = state
        .http
        .get(url.clone())
        .timeout(PROBE_TIMEOUT)
        .send()
        .await
        .and_then(|r| r.error_for_status());

    match result {
        Ok(_) => Dependency {
            url: url.to_string(),
            reachable: true,
            latency_ms: Some(started.elapsed().as_millis()),
            error: None,
        },
        Err(e) => Dependency {
            url: url.to_string(),
            reachable: false,
            latency_ms: None,
            error: Some(e.to_string()),
        },
    }
}
