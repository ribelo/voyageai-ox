use std::{
    borrow::{Borrow, Cow},
    sync::Arc,
};

#[cfg(feature = "leaky-bucket")]
use leaky_bucket::RateLimiter;
use serde::{Deserialize, Serialize};
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
    #[error("JSON serialization/deserialization error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("HTTP request error: {0}")]
    RequestError(#[from] reqwest::Error),
}

#[derive(Error, Debug)]
pub enum VoyageBuilderError {
    #[error("API key not set")]
    ApiKeyNotSet,
}

#[derive(Debug, Default)]
pub struct VoyageBuilder {
    api_key: Option<String>,
    client: Option<reqwest::Client>,
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

    pub fn client(mut self, client: reqwest::Client) -> Self {
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
    client: reqwest::Client,
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
pub enum EmbeddingsInput<'a> {
    Single(Cow<'a, str>),
    Multiple(Vec<Cow<'a, str>>),
}

impl<'a> From<&'a str> for EmbeddingsInput<'a> {
    fn from(s: &'a str) -> Self {
        EmbeddingsInput::Single(Cow::Borrowed(s))
    }
}

impl<'a> From<Vec<&'a str>> for EmbeddingsInput<'a> {
    fn from(v: Vec<&'a str>) -> Self {
        EmbeddingsInput::Multiple(v.into_iter().map(Cow::Borrowed).collect())
    }
}

impl<'a> From<String> for EmbeddingsInput<'a> {
    fn from(s: String) -> Self {
        EmbeddingsInput::Single(Cow::Owned(s))
    }
}

impl<'a> From<Vec<String>> for EmbeddingsInput<'a> {
    fn from(v: Vec<String>) -> Self {
        EmbeddingsInput::Multiple(v.into_iter().map(Cow::Owned).collect())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Base64,
}

#[derive(Debug, Default)]
pub struct EmbeddingsRequestBuilder<'a> {
    input: Option<EmbeddingsInput<'a>>,
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

impl<'a> EmbeddingsRequestBuilder<'a> {
    pub fn input<T: Into<EmbeddingsInput<'a>>>(mut self, input: T) -> Self {
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

    pub fn build(self) -> Result<EmbeddingsRequest<'a>, EmbeddingsBuilderError> {
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
pub struct EmbeddingsRequest<'a> {
    pub input: EmbeddingsInput<'a>,
    pub model: EmbeddingModel,
    pub input_type: Option<InputType>,
    #[serde(skip)]
    pub voyage: Voyage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
}

impl<'a> EmbeddingsRequest<'a> {
    pub fn builder() -> EmbeddingsRequestBuilder<'a> {
        EmbeddingsRequestBuilder::default()
    }

    pub async fn send(&self) -> Result<EmbeddingsResponse, VoyageError> {
        #[cfg(feature = "leaky-bucket")]
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
pub struct RerankRequestBuilder<'a> {
    query: Option<String>,
    documents: Option<Vec<Cow<'a, str>>>,
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

impl<'a> RerankRequestBuilder<'a> {
    pub fn query<T: Into<String>>(mut self, query: T) -> Self {
        self.query = Some(query.into());
        self
    }

    pub fn documents<I, S>(mut self, documents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Cow<'a, str>>,
    {
        self.documents = Some(documents.into_iter().map(Into::into).collect());
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

    pub fn build(self) -> Result<RerankRequest<'a>, RerankBuilderError> {
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
pub struct RerankRequest<'a> {
    query: String,
    documents: Vec<Cow<'a, str>>,
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

impl<'a> RerankRequest<'a> {
    pub fn builder() -> RerankRequestBuilder<'a> {
        RerankRequestBuilder::default()
    }

    pub async fn send(self) -> Result<RerankResponse, VoyageError> {
        #[cfg(feature = "leaky-bucket")]
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

    // #[cfg(target_arch = "wasm32")]
    // pub async fn send(self) -> Result<RerankResponse, VoyageError> {
    //     use web_sys::{Headers, RequestMode};
    //
    //     #[cfg(feature = "leaky-bucket")]
    //     if let Some(limiter) = self.voyage.leaky_bucket.as_ref() {
    //         limiter.acquire_one().await
    //     }
    //     let url = format!("{}/rerank", BASE_URL);
    //
    //     let mut opts = RequestInit::new();
    //     opts.method("POST");
    //     opts.mode(RequestMode::Cors);
    //
    //     let headers = Headers::new()
    //         .map_err(|e| VoyageError::HeaderCreationError(e.as_string().unwrap_or_default()))?;
    //     headers
    //         .append("Authorization", &format!("Bearer {}", self.voyage.api_key))
    //         .map_err(|e| VoyageError::HeaderAppendError(e.as_string().unwrap_or_default()))?;
    //     headers
    //         .append("Content-Type", "application/json")
    //         .map_err(|e| VoyageError::HeaderAppendError(e.as_string().unwrap_or_default()))?;
    //     opts.headers(&headers);
    //
    //     let body = serde_json::to_string(&self)?;
    //     opts.body(Some(&JsValue::from(body)));
    //
    //     let request = Request::new_with_str_and_init(&url, &opts)
    //         .map_err(|e| VoyageError::RequestCreationError(e.as_string().unwrap_or_default()))?;
    //
    //     let window = web_sys::window().unwrap();
    //     let resp_value = JsFuture::from(window.fetch_with_request(&request))
    //         .await
    //         .map_err(|e| VoyageError::FetchError(e.as_string().unwrap_or_default()))?;
    //     let resp: Response = resp_value.dyn_into().unwrap();
    //
    //     match resp.status() {
    //         200 => {
    //             let json = JsFuture::from(resp.json().unwrap()).await.map_err(|e| {
    //                 VoyageError::ResponseParseError(e.as_string().unwrap_or_default())
    //             })?;
    //             let result: RerankResponse = serde_wasm_bindgen::from_value(json)?;
    //             Ok(result)
    //         }
    //         400 => Err(VoyageError::InvalidRequest {
    //             message: JsFuture::from(resp.text().unwrap())
    //                 .await
    //                 .map_err(|e| {
    //                     VoyageError::ResponseParseError(e.as_string().unwrap_or_default())
    //                 })?
    //                 .as_string()
    //                 .unwrap(),
    //         }),
    //         401 => Err(VoyageError::Unauthorized),
    //         429 => Err(VoyageError::RateLimitExceeded),
    //         500 => Err(VoyageError::ServerError(
    //             JsFuture::from(resp.text().unwrap())
    //                 .await
    //                 .map_err(|e| {
    //                     VoyageError::ResponseParseError(e.as_string().unwrap_or_default())
    //                 })?
    //                 .as_string()
    //                 .unwrap(),
    //         )),
    //         _ => Err(VoyageError::ServiceUnavailable),
    //     }
    // }
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

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test_configure!(run_in_browser);

    #[cfg(not(target_arch = "wasm32"))]
    fn get_api_key() -> String {
        env::var("VOYAGE_API_KEY").expect("VOYAGE_API_KEY must be set")
    }

    #[cfg(target_arch = "wasm32")]
    const fn get_api_key() -> &'static str {
        env!("VOYAGE_API_KEY")
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_voyage_builder() {
        let api_key = get_api_key();
        let voyage = Voyage::builder()
            .api_key(api_key)
            .build()
            .expect("Failed to build Voyage");

        assert!(!voyage.api_key.is_empty());
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_embeddings_request_multiple_inputs() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();
        let docs = vec!["Input 1".to_string(), "Input 2".to_string()];

        let request = voyage
            .embeddings()
            .input(docs)
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    async fn test_rerank_request() {
        let api_key = get_api_key();
        let voyage = Voyage::builder().api_key(api_key).build().unwrap();
        let docs = vec![
            "Document 1".to_string(),
            "Document 2".to_string(),
            "Document 3".to_string(),
        ];

        let request = voyage
            .rerank()
            .query("Test query")
            .documents(&docs)
            .model(RerankerModel::RerankLite1)
            .build()
            .expect("Failed to build rerank request");

        let response = request.send().await.expect("Failed to send rerank request");

        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 3);
        assert_eq!(response.model, RerankerModel::RerankLite1);
        assert!(response.usage.total_tokens > 0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

        let array = EmbeddingsInput::from(vec!["test1".to_string(), "test2".to_string()]);
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

    #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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
