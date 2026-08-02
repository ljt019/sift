// src/api/search.rs
use axum::Json;
use axum::extract::State;

use super::types::{SearchRequest, SearchResponse, SearchResult};
use crate::error::{AppError, Result};
use crate::search::{self, Params};
use crate::state::AppState;

const MAX_RESULTS: usize = 10;
const DEFAULT_CONTEXT_MAX_CHARACTERS: usize = 24_000;
const MAX_CONTEXT_MAX_CHARACTERS: usize = 100_000;

pub async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>> {
    let query = req.query.trim();
    if query.is_empty() {
        return Err(AppError::BadRequest("query must not be empty".into()));
    }
    if req.num_results == 0 || req.num_results > MAX_RESULTS {
        return Err(AppError::BadRequest(format!(
            "numResults must be between 1 and {MAX_RESULTS}"
        )));
    }

    let context_max_characters = req
        .context_max_characters
        .unwrap_or(DEFAULT_CONTEXT_MAX_CHARACTERS);
    if context_max_characters == 0 || context_max_characters > MAX_CONTEXT_MAX_CHARACTERS {
        return Err(AppError::BadRequest(format!(
            "contextMaxCharacters must be between 1 and {MAX_CONTEXT_MAX_CHARACTERS}"
        )));
    }

    let params = Params::new(req.num_results, context_max_characters);
    let results = search::run(&state, query, &params)
        .await?
        .into_iter()
        .map(SearchResult::from)
        .collect();

    Ok(Json(SearchResponse { results }))
}
