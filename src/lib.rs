use std::sync::Arc;

use leaky_bucket::RateLimiter;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use thiserror::Error;

const BASE_URL: &str = "https://api.voyageai.com/v1";

#[derive(Debug, Error)]
pub enum VoyageError {
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("Unauthorized: Invalid API key")]
    Unauthorized,
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Server error: {0}")]
    ServerError(String),
    #[error("Service unavailable")]
    ServiceUnavailable,
    #[error("HTTP request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("JSON serialization/deserialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
}

#[derive(Error, Debug)]
pub enum VoyageBuilderError {
    #[error("API key not set")]
    ApiKeyNotSet,
}

#[derive(Debug, Default)]
pub struct VoyageBuilder {
    api_key: Option<String>,
    client: Option<Client>,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<RateLimiter>,
}

impl VoyageBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn api_key<T: Into<String>>(mut self, api_key: T) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    #[cfg(feature = "leaky-bucket")]
    pub fn leaky_bucket(mut self, leaky_bucket: RateLimiter) -> Self {
        self.leaky_bucket = Some(leaky_bucket);
        self
    }

    pub fn build(self) -> Result<Voyage, VoyageBuilderError> {
        let api_key = self.api_key.ok_or(VoyageBuilderError::ApiKeyNotSet)?;
        let client = self.client.unwrap_or_default();

        #[cfg(feature = "leaky-bucket")]
        let leaky_bucket = self.leaky_bucket.map(Arc::new);

        Ok(Voyage {
            api_key,
            client,
            #[cfg(feature = "leaky-bucket")]
            leaky_bucket,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Voyage {
    api_key: String,
    client: Client,
    #[cfg(feature = "leaky-bucket")]
    leaky_bucket: Option<Arc<RateLimiter>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingModel {
    #[serde(rename = "voyage-large-2-instruct")]
    VoyageLarge2Instruct,
    #[serde(rename = "voyage-finance-2")]
    VoyageFinance2,
    #[serde(rename = "voyage-multilingual-2")]
    VoyageMultilingual2,
    #[serde(rename = "voyage-law-2")]
    VoyageLaw2,
    #[serde(rename = "voyage-code-2")]
    VoyageCode2,
    #[serde(rename = "voyage-large-2")]
    VoyageLarge2,
    #[serde(rename = "voyage-2")]
    Voyage2,
}

impl EmbeddingModel {
    pub fn context_length(&self) -> usize {
        match self {
            Self::VoyageLarge2Instruct => 16000,
            Self::VoyageFinance2 => 32000,
            Self::VoyageMultilingual2 => 32000,
            Self::VoyageLaw2 => 16000,
            Self::VoyageCode2 => 16000,
            Self::VoyageLarge2 => 16000,
            Self::Voyage2 => 4000,
        }
    }

    pub fn embedding_dimension(&self) -> usize {
        match self {
            Self::VoyageLarge2Instruct => 1024,
            Self::VoyageFinance2 => 1024,
            Self::VoyageMultilingual2 => 1024,
            Self::VoyageLaw2 => 1024,
            Self::VoyageCode2 => 1536,
            Self::VoyageLarge2 => 1536,
            Self::Voyage2 => 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    Query,
    Document,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Multiple(Vec<String>),
}

impl From<String> for EmbeddingsInput {
    fn from(s: String) -> Self {
        EmbeddingsInput::Single(s)
    }
}

impl From<&str> for EmbeddingsInput {
    fn from(s: &str) -> Self {
        EmbeddingsInput::Single(s.to_string())
    }
}

impl From<Vec<String>> for EmbeddingsInput {
    fn from(v: Vec<String>) -> Self {
        EmbeddingsInput::Multiple(v)
    }
}

impl<const N: usize> From<[String; N]> for EmbeddingsInput {
    fn from(arr: [String; N]) -> Self {
        EmbeddingsInput::Multiple(arr.to_vec())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Base64,
}

#[derive(Debug, Default)]
pub struct EmbeddingsRequestBuilder {
    input: Option<EmbeddingsInput>,
    model: Option<EmbeddingModel>,
    input_type: Option<InputType>,
    voyage: Option<Voyage>,
    truncation: Option<bool>,
    encoding_format: Option<EncodingFormat>,
}

#[derive(Debug, Error)]
pub enum EmbeddingsBuilderError {
    #[error("Missing input field")]
    MissingInput,
    #[error("Missing model field")]
    MissingModel,
    #[error("Missing voyage field")]
    MissingVoyage,
    #[error("Input list exceeds maximum length of 128")]
    InputListTooLong,
}

impl EmbeddingsRequestBuilder {
    pub fn input(mut self, input: impl Into<EmbeddingsInput>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn model(mut self, model: EmbeddingModel) -> Self {
        self.model = Some(model);
        self
    }

    pub fn input_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    pub fn voyage(mut self, voyage: Voyage) -> Self {
        self.voyage = Some(voyage);
        self
    }

    pub fn truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }

    pub fn encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    pub fn build(self) -> Result<EmbeddingsRequest, EmbeddingsBuilderError> {
        let input = self.input.ok_or(EmbeddingsBuilderError::MissingInput)?;
        let model = self.model.ok_or(EmbeddingsBuilderError::MissingModel)?;
        let voyage = self.voyage.ok_or(EmbeddingsBuilderError::MissingVoyage)?;

        if let EmbeddingsInput::Multiple(ref texts) = input {
            if texts.len() > 128 {
                return Err(EmbeddingsBuilderError::InputListTooLong);
            }
        }

        Ok(EmbeddingsRequest {
            input,
            model,
            input_type: self.input_type,
            voyage,
            truncation: self.truncation,
            encoding_format: self.encoding_format,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingsInput,
    pub model: EmbeddingModel,
    pub input_type: Option<InputType>,
    #[serde(skip)]
    pub voyage: Voyage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
}

impl EmbeddingsRequest {
    pub fn builder() -> EmbeddingsRequestBuilder {
        EmbeddingsRequestBuilder::default()
    }

    pub async fn send(&self) -> Result<EmbeddingsResponse, VoyageError> {
        if let Some(limiter) = self.voyage.leaky_bucket.as_ref() {
            limiter.acquire_one().await
        }
        let url = format!("{}/embeddings", BASE_URL);

        let response = self
            .voyage
            .client
            .post(&url)
            .bearer_auth(&self.voyage.api_key)
            .json(self)
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(response.json().await?),
            reqwest::StatusCode::BAD_REQUEST => Err(VoyageError::InvalidRequest {
                message: response.text().await?,
            }),
            reqwest::StatusCode::UNAUTHORIZED => Err(VoyageError::Unauthorized),
            reqwest::StatusCode::TOO_MANY_REQUESTS => Err(VoyageError::RateLimitExceeded),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                Err(VoyageError::ServerError(response.text().await?))
            }
            _ => Err(VoyageError::ServiceUnavailable),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: EmbeddingModel,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RerankerModel {
    #[serde(rename = "rerank-1")]
    Rerank1,
    #[serde(rename = "rerank-lite-1")]
    RerankLite1,
}

impl RerankerModel {
    pub fn context_length(&self) -> usize {
        match self {
            Self::Rerank1 => 8000,
            Self::RerankLite1 => 4000,
        }
    }
}

#[derive(Debug, Default)]
pub struct RerankRequestBuilder {
    query: Option<String>,
    documents: Option<Vec<String>>,
    model: Option<RerankerModel>,
    voyage: Option<Voyage>,
    top_k: Option<u32>,
    return_documents: Option<bool>,
    truncation: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum RerankBuilderError {
    #[error("Missing query field")]
    MissingQuery,
    #[error("Missing documents field")]
    MissingDocuments,
    #[error("Missing model field")]
    MissingModel,
    #[error("Missing voyage field")]
    MissingVoyage,
}

impl RerankRequestBuilder {
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    pub fn documents(mut self, documents: Vec<String>) -> Self {
        self.documents = Some(documents);
        self
    }

    pub fn model(mut self, model: RerankerModel) -> Self {
        self.model = Some(model);
        self
    }

    pub fn voyage(mut self, voyage: Voyage) -> Self {
        self.voyage = Some(voyage);
        self
    }

    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn return_documents(mut self) -> Self {
        self.return_documents = Some(true);
        self
    }

    pub fn without_truncation(mut self) -> Self {
        self.truncation = Some(false);
        self
    }

    pub fn build(self) -> Result<RerankRequest, RerankBuilderError> {
        let query = self.query.ok_or(RerankBuilderError::MissingQuery)?;
        let documents = self.documents.ok_or(RerankBuilderError::MissingDocuments)?;
        let model = self.model.ok_or(RerankBuilderError::MissingModel)?;
        let voyage = self.voyage.ok_or(RerankBuilderError::MissingVoyage)?;

        Ok(RerankRequest {
            query,
            documents,
            model,
            voyage,
            top_k: self.top_k,
            return_documents: self.return_documents,
            truncation: self.truncation,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct RerankRequest {
    query: String,
    documents: Vec<String>,
    model: RerankerModel,
    #[serde(skip)]
    voyage: Voyage,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<bool>,
}

impl RerankRequest {
    pub fn builder() -> RerankRequestBuilder {
        RerankRequestBuilder::default()
    }

    pub async fn send(self) -> Result<RerankResponse, VoyageError> {
        if let Some(limiter) = self.voyage.leaky_bucket.as_ref() {
            limiter.acquire_one().await
        }
        let url = format!("{}/rerank", BASE_URL);
        let response = self
            .voyage
            .client
            .post(&url)
            .bearer_auth(&self.voyage.api_key)
            .json(&self)
            .send()
            .await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(response.json().await?),
            reqwest::StatusCode::BAD_REQUEST => Err(VoyageError::InvalidRequest {
                message: response.text().await?,
            }),
            reqwest::StatusCode::UNAUTHORIZED => Err(VoyageError::Unauthorized),
            reqwest::StatusCode::TOO_MANY_REQUESTS => Err(VoyageError::RateLimitExceeded),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                Err(VoyageError::ServerError(response.text().await?))
            }
            _ => Err(VoyageError::ServiceUnavailable),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub object: String,
    pub data: Vec<RerankResult>,
    pub model: RerankerModel,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct RerankResult {
    pub relevance_score: f32,
    pub index: usize,
}

impl Voyage {
    pub fn builder() -> VoyageBuilder {
        VoyageBuilder::default()
    }

    pub fn embeddings(&self) -> EmbeddingsRequestBuilder {
        EmbeddingsRequest::builder().voyage(self.clone())
    }

    pub fn rerank(&self) -> RerankRequestBuilder {
        RerankRequest::builder().voyage(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // Helper function to get API key from environment
    fn get_api_key() -> String {
        env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set")
    }

    #[tokio::test]
    async fn test_voyage_builder() {
        let api_key = get_api_key();
        let voyage = Voyage::builder()
            .api_key(api_key)
            .build()
            .expect("Failed to build Voyage");

        assert!(!voyage.api_key.is_empty());
    }

    #[tokio::test]
    async fn test_embeddings_request() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();

        let request = voyage
            .embeddings()
            .input("Test input")
            .model(EmbeddingModel::VoyageLarge2Instruct)
            .build()
            .expect("Failed to build embeddings request");

        let response = request
            .send()
            .await
            .expect("Failed to send embeddings request");

        assert_eq!(response.object, "list");
        assert!(!response.data.is_empty());
        assert_eq!(response.model, EmbeddingModel::VoyageLarge2Instruct);
        assert!(response.usage.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_embeddings_request_multiple_inputs() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();

        let request = voyage
            .embeddings()
            .input(vec!["Input 1".to_string(), "Input 2".to_string()])
            .model(EmbeddingModel::VoyageLarge2Instruct)
            .build()
            .expect("Failed to build embeddings request");

        let response = request
            .send()
            .await
            .expect("Failed to send embeddings request");

        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.model, EmbeddingModel::VoyageLarge2Instruct);
        assert!(response.usage.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_embeddings_request_with_options() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();

        let request = voyage
            .embeddings()
            .input("Test input")
            .model(EmbeddingModel::VoyageLarge2Instruct)
            .input_type(InputType::Query)
            .truncation(true)
            .build()
            .expect("Failed to build embeddings request");

        let response = request
            .send()
            .await
            .expect("Failed to send embeddings request");

        assert_eq!(response.object, "list");
        assert!(!response.data.is_empty());
        assert_eq!(response.model, EmbeddingModel::VoyageLarge2Instruct);
        assert!(response.usage.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_rerank_request() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();

        let request = voyage
            .rerank()
            .query("Test query")
            .documents(vec![
                "Document 1".to_string(),
                "Document 2".to_string(),
                "Document 3".to_string(),
            ])
            .model(RerankerModel::RerankLite1)
            .build()
            .expect("Failed to build rerank request");

        let response = request.send().await.expect("Failed to send rerank request");

        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 3);
        assert_eq!(response.model, RerankerModel::RerankLite1);
        assert!(response.usage.total_tokens > 0);
    }

    #[tokio::test]
    async fn test_rerank_request_with_options() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();

        let request = voyage
            .rerank()
            .query("Test query")
            .documents(vec![
                "Document 1".to_string(),
                "Document 2".to_string(),
                "Document 3".to_string(),
            ])
            .model(RerankerModel::RerankLite1)
            .top_k(2)
            .return_documents()
            .without_truncation()
            .build()
            .expect("Failed to build rerank request");

        let response = request.send().await.expect("Failed to send rerank request");

        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2); // top_k is 2
        assert_eq!(response.model, RerankerModel::RerankLite1);
        assert!(response.usage.total_tokens > 0);
    }

    #[test]
    fn test_embeddings_input_from() {
        let single = EmbeddingsInput::from("test");
        assert!(matches!(single, EmbeddingsInput::Single(_)));

        let multiple = EmbeddingsInput::from(vec!["test1".to_string(), "test2".to_string()]);
        assert!(matches!(multiple, EmbeddingsInput::Multiple(_)));

        let array = EmbeddingsInput::from(["test1".to_string(), "test2".to_string()]);
        assert!(matches!(array, EmbeddingsInput::Multiple(_)));
    }

    #[test]
    #[should_panic(expected = "InputListTooLong")]
    fn test_embeddings_request_too_many_inputs() {
        let voyage = Voyage::builder().api_key("test").build().unwrap();
        let inputs: Vec<String> = (0..129).map(|i| format!("Input {}", i)).collect();

        voyage
            .embeddings()
            .input(inputs)
            .model(EmbeddingModel::VoyageLarge2Instruct)
            .build()
            .unwrap();
    }

    #[tokio::test]
    async fn test_invalid_api_key() {
        let voyage = Voyage::builder().api_key("invalid_key").build().unwrap();

        let request = voyage
            .embeddings()
            .input("Test input")
            .model(EmbeddingModel::VoyageLarge2Instruct)
            .build()
            .unwrap();

        let result = request.send().await;
        assert!(matches!(result, Err(VoyageError::Unauthorized)));
    }
}
