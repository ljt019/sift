use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("search backend unavailable: {0}")]
    SearchBackend(#[source] reqwest::Error),

    #[error("search backend returned an unexpected response: {0}")]
    SearchBackendResponse(String),

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    fn status(&self) -> StatusCode {
        match self {
            Self::SearchBackend(error) if error.is_timeout() => StatusCode::GATEWAY_TIMEOUT,
            Self::SearchBackend(_) | Self::SearchBackendResponse(_) => StatusCode::BAD_GATEWAY,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::SearchBackend(_) | Self::SearchBackendResponse(_) => "search_backend",
            Self::BadRequest(_) => "bad_request",
            Self::Internal(_) => "internal",
        }
    }

    fn client_message(&self) -> String {
        match self {
            Self::SearchBackend(_) => "search backend unavailable".to_string(),
            Self::SearchBackendResponse(_) => {
                "search backend returned an unexpected response".to_string()
            }
            Self::BadRequest(message) => message.clone(),
            Self::Internal(_) => "internal server error".to_string(),
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: &'static str,
    message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();

        if status.is_server_error() {
            match &self {
                Self::Internal(error) => tracing::error!(error = ?error, "request failed"),
                other => tracing::error!(error = %other, "request failed"),
            }
        } else {
            tracing::debug!(error = %self, "request rejected");
        }

        let message = self.client_message();

        (
            status,
            Json(ErrorBody {
                error: self.kind(),
                message,
            }),
        )
            .into_response()
    }
}

pub type Result<T> = std::result::Result<T, AppError>;

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;

    use super::*;

    #[test]
    fn backend_response_detail_is_redacted() {
        let error = AppError::SearchBackendResponse("secret upstream response".to_string());

        assert_eq!(error.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(error.kind(), "search_backend");
        assert_eq!(
            error.client_message(),
            "search backend returned an unexpected response"
        );
    }

    #[test]
    fn internal_error_detail_is_redacted() {
        let error = AppError::Internal(anyhow::anyhow!("secret internal context"));

        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.kind(), "internal");
        assert_eq!(error.client_message(), "internal server error");
    }

    #[tokio::test]
    async fn internal_error_response_is_redacted() {
        let response =
            AppError::Internal(anyhow::anyhow!("secret internal context")).into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("error response body should be readable");
        let body: serde_json::Value =
            serde_json::from_slice(&body).expect("error response body should be valid JSON");

        assert_eq!(body["error"], "internal");
        assert_eq!(body["message"], "internal server error");
        assert!(!body.to_string().contains("secret internal context"));
    }
}
