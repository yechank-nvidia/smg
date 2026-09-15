//! Exercise the completion processor with real protobuf frames and stop decoding.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::{BufMut, Bytes};
use http_body::Frame;
use http_body_util::StreamBody;
use llm_tokenizer::MockTokenizer;
use openai_protocol::completion::CompletionRequest;
use prost::Message as ProstMessage;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{
    proto, proto::generate_response::Response as GenerationEvent, AbortOnDropStream,
    VllmEngineClient,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tonic::codec::Codec;

use super::*;
use crate::{routers::grpc::proto_wrapper::ProtoStream, worker::WorkerRegistry};

/// The mock server accepts the client connection. This test supplies the
/// response frames and controls EOF.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; server task is explicitly aborted"
)]
async fn scripted_stream(
    responses: Vec<proto::GenerateResponse>,
    grpc_status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock worker");
    let port = listener.local_addr().expect("mock worker address").port();
    let config = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: "completion-logprobs-test".to_string(),
        tokenizer_path: "completion-logprobs-test".to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 0,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    let client = VllmEngineClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect mock worker");
    let mut frames = Vec::new();
    for response in responses {
        let encoded = response.encode_to_vec();
        let mut frame = Vec::with_capacity(encoded.len() + 5);
        frame.put_u8(0);
        frame.put_u32(encoded.len() as u32);
        frame.extend_from_slice(&encoded);
        frames.push(Ok::<_, tonic::Status>(Frame::data(Bytes::from(frame))));
    }
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", HeaderValue::from_static(grpc_status));
    frames.push(Ok(Frame::trailers(trailers)));
    let mut codec =
        tonic_prost::ProstCodec::<proto::GenerateResponse, proto::GenerateResponse>::default();
    let stream = tonic::Streaming::new_response(
        codec.decoder(),
        StreamBody::new(futures::stream::iter(frames)),
        StatusCode::OK,
        None,
        None,
    );
    let stream = AbortOnDropStream::new(stream, "completion-logprobs-test".to_string(), client);
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    (ProtoStream::Vllm(stream), server)
}

fn complete(logprobs: Option<proto::OutputLogProbs>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            output_ids: vec![1, 2],
            output_logprobs: logprobs,
            prompt_tokens: 3,
            completion_tokens: 2,
            finish_reason: "length".to_string(),
            ..Default::default()
        })),
    }
}

fn logprobs() -> proto::OutputLogProbs {
    proto::OutputLogProbs {
        token_ids: vec![1, 2],
        token_logprobs: vec![-0.25, -1.5],
        top_logprobs: vec![],
    }
}

async fn process(
    request: Value,
    groups: Vec<Vec<proto::GenerateResponse>>,
) -> Result<CompletionResponse, axum::response::Response> {
    let request: CompletionRequest = serde_json::from_value(request).unwrap();
    let spec = CompletionResponseSpec::from(&request);
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let mut stop_decoder = utils::create_stop_decoder(
        &tokenizer,
        request.stop.as_ref(),
        request.stop_token_ids.as_ref(),
        request.skip_special_tokens,
        request.no_stop_trim,
        request.ignore_eos,
    );
    let mut servers = Vec::new();
    let mut results = Vec::new();
    for frames in groups {
        let (stream, server) = scripted_stream(frames, "0").await;
        servers.push(server);
        results.push(ExecutionResult::Single { stream });
    }
    let result = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::new(Arc::new(WorkerRegistry::new()), None, None),
    )
    .process_non_streaming_completion_response(
        ExecutionResult::Batch { results },
        spec,
        DispatchMetadata {
            request_id: "completion-logprobs-test".into(),
            model: "test-model".into(),
            created: 1,
            weight_version: None,
        },
        tokenizer,
        &mut stop_decoder,
    )
    .await;
    for server in servers {
        server.abort();
    }
    result
}

#[tokio::test]
async fn batched_echo_choices_keep_logprobs_and_character_offsets() {
    let output = complete(Some(logprobs()));
    let response = process(
        json!({
            "model": "test-model", "prompt": ["é", "한국어"], "echo": true,
            "suffix": "!", "n": 2, "logprobs": 0
        }),
        vec![
            vec![output.clone(), output.clone()],
            vec![output.clone(), output],
        ],
    )
    .await
    .unwrap();
    assert_eq!(response.choices.len(), 4);
    for (index, choice) in response.choices.iter().enumerate() {
        assert_eq!(choice.index, index as u32);
        let (prompt, offset) = if index < 2 {
            ("é", 1)
        } else {
            ("한국어", 3)
        };
        assert_eq!(choice.text, format!("{prompt}Hello world!"));
        let scores = choice.logprobs.as_ref().unwrap();
        assert_eq!(scores.tokens, ["Hello", " world"]);
        assert_eq!(scores.text_offset, [offset, offset + 5]);
        assert_eq!(scores.token_logprobs, [Some(-0.25), Some(-1.5)]);
    }
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 6);
    assert_eq!(usage.completion_tokens, 8);
}

#[tokio::test]
async fn stop_trimming_and_top_candidates_match_visible_text() {
    let mut scores = logprobs();
    scores.top_logprobs = vec![
        proto::TopLogProbs::default(),
        proto::TopLogProbs {
            values: vec![-1.5, -2.0],
            token_ids: vec![2, 3],
        },
    ];
    let response = process(
        json!({
            "model": "test-model", "prompt": "Hello", "stop": "world", "logprobs": 2
        }),
        vec![vec![complete(Some(scores))]],
    )
    .await
    .unwrap();
    let choice = &response.choices[0];
    assert_eq!(choice.text, "Hello ");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    let scores = choice.logprobs.as_ref().unwrap();
    assert_eq!(scores.tokens, ["Hello", " "]);
    assert_eq!(scores.text_offset, [0, 5]);
    assert_eq!(scores.top_logprobs[1].as_ref().unwrap()["test"], -2.0);
}

#[tokio::test]
async fn requested_missing_or_mismatched_scores_fail_but_unrequested_are_ignored() {
    let mut wrong = logprobs();
    wrong.token_ids[0] = 3;
    for payload in [None, Some(wrong)] {
        let error = process(
            json!({
                "model": "test-model", "prompt": "Hello", "logprobs": 0
            }),
            vec![vec![complete(payload.clone())]],
        )
        .await
        .unwrap_err();
        assert_eq!(error.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(error.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "completion_logprobs_failed");
        let response = process(
            json!({
                "model": "test-model", "prompt": "Hello"
            }),
            vec![vec![complete(payload)]],
        )
        .await
        .unwrap();
        assert_eq!(response.choices[0].text, "Hello world");
        assert!(response.choices[0].logprobs.is_none());
    }
}

#[tokio::test]
async fn empty_completion_has_empty_requested_logprobs() {
    let output = proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            finish_reason: "length".into(),
            ..Default::default()
        })),
    };
    let response = process(
        json!({
            "model": "test-model", "prompt": "Hello", "echo": true,
            "max_tokens": 0, "logprobs": 0
        }),
        vec![vec![output]],
    )
    .await
    .unwrap();
    let choice = &response.choices[0];
    assert_eq!(choice.text, "Hello");
    assert_eq!(
        serde_json::to_value(choice.logprobs.as_ref().unwrap()).unwrap(),
        json!({
            "tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []
        })
    );
}
