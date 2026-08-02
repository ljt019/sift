use std::time::Duration;

use axum::routing::{get, post};
use axum::{Router, http::StatusCode};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

mod health;
mod search;
mod types;

pub use types::{Dependency, HealthResponse, Status};

/// Outer backstop only. Dependency-specific budgets are enforced closer to
/// their operations; this exists so a hung request can't leak a connection.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub fn build(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/search", post(search::search))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            REQUEST_TIMEOUT,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
