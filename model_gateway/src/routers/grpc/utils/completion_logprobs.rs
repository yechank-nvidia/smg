//! Conversion of generated-token logprobs to the legacy Completions wire shape.

use std::{collections::HashMap, sync::Arc};

use llm_tokenizer::{traits::Tokenizer, DecodeStream};
use openai_protocol::common::LogProbs;

use crate::routers::grpc::proto_wrapper::ProtoOutputLogProbs;

/// Preserve contextual token decoding and align scores with the visible output.
/// `text_offset` is the number of characters prepended by prompt echo. Literal
/// suffixes and echoed prompt text do not receive generated-token scores.
pub(crate) fn convert_completion_logprobs(
    proto: &ProtoOutputLogProbs,
    output_ids: &[u32],
    tokenizer: Arc<dyn Tokenizer>,
    skip_special_tokens: bool,
    output_text: &str,
    text_offset: usize,
) -> Result<LogProbs, String> {
    validate_payload(proto, output_ids)?;
    let expected = tokenizer
        .decode(output_ids, skip_special_tokens)
        .map_err(|error| format!("Failed to decode completion output tokens: {error}"))?;
    let mut stream = DecodeStream::new(tokenizer.clone(), &[], skip_special_tokens);
    let mut tokens = Vec::with_capacity(output_ids.len());
    for &id in output_ids {
        tokens.push(
            stream
                .step(id)
                .map_err(|error| format!("Failed to decode completion token {id}: {error}"))?
                .unwrap_or_default(),
        );
    }
    if let Some(tail) = stream
        .flush()
        .map_err(|error| format!("Failed to flush completion token decoder: {error}"))?
    {
        tokens
            .last_mut()
            .ok_or_else(|| "Completion decoder flushed text for empty output".to_string())?
            .push_str(&tail);
    }
    if tokens.concat() != expected {
        return Err("Incremental completion decode does not match full decode".to_string());
    }
    build_logprobs(proto, tokens, output_text, text_offset, |id| {
        tokenizer
            .decode(&[id], skip_special_tokens)
            .map_err(|error| format!("Failed to decode completion candidate {id}: {error}"))
    })
}

fn validate_payload(proto: &ProtoOutputLogProbs, output_ids: &[u32]) -> Result<(), String> {
    if proto.token_ids != output_ids || proto.token_logprobs.len() != output_ids.len() {
        return Err("Completion logprob token ids or counts do not match output ids".to_string());
    }
    if !proto.top_logprobs.is_empty() && proto.top_logprobs.len() != output_ids.len() {
        return Err("Completion top-logprob count does not match output ids".to_string());
    }
    if proto.token_logprobs.iter().any(|value| !value.is_finite()) {
        return Err("Completion token logprob is not finite".to_string());
    }
    for top in &proto.top_logprobs {
        if top.token_ids.len() != top.values.len() {
            return Err("Completion candidate token and logprob counts do not match".to_string());
        }
        if top.values.iter().any(|value| !value.is_finite()) {
            return Err("Completion candidate logprob is not finite".to_string());
        }
    }
    Ok(())
}

fn build_logprobs(
    proto: &ProtoOutputLogProbs,
    tokens: Vec<String>,
    output_text: &str,
    mut offset: usize,
    decode_candidate: impl Fn(u32) -> Result<String, String>,
) -> Result<LogProbs, String> {
    // Gateway-side stop matching may remove a suffix, including a stop string
    // inside a token. Keep only token fragments present in the returned text.
    if !tokens.concat().starts_with(output_text) {
        return Err("Completion text does not match decoded logprob tokens".to_string());
    }
    let mut remaining = output_text;
    let mut result = LogProbs {
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        top_logprobs: Vec::new(),
        text_offset: Vec::new(),
    };
    for (index, mut token) in tokens.into_iter().enumerate() {
        if remaining.is_empty() {
            break;
        }
        token.truncate(token.len().min(remaining.len()));
        remaining = &remaining[token.len()..];
        result.text_offset.push(
            u32::try_from(offset)
                .map_err(|_| "Completion text offset does not fit in u32".to_string())?,
        );
        offset = offset
            .checked_add(token.chars().count())
            .ok_or_else(|| "Completion text offset overflowed usize".to_string())?;
        let sampled = proto.token_logprobs[index];
        let mut top_logprobs: HashMap<String, f32> = HashMap::new();
        if let Some(top) = proto.top_logprobs.get(index) {
            for (&id, &value) in top.token_ids.iter().zip(&top.values) {
                if id == proto.token_ids[index] {
                    continue;
                }
                top_logprobs
                    .entry(decode_candidate(id)?)
                    .and_modify(|previous| *previous = previous.max(value))
                    .or_insert(value);
            }
        }
        // Include the sampled token even when the backend supplied no top-k
        // alternatives (logprobs=0), or supplied empty per-position placeholders.
        top_logprobs.insert(token.clone(), sampled);
        result.tokens.push(token);
        result.token_logprobs.push(Some(sampled));
        result.top_logprobs.push(Some(top_logprobs));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::MockTokenizer;

    use super::*;
    use crate::routers::grpc::proto_wrapper::ProtoTopLogProbs;

    fn payload() -> ProtoOutputLogProbs {
        ProtoOutputLogProbs {
            token_ids: vec![1, 2],
            token_logprobs: vec![-0.25, -1.5],
            top_logprobs: Vec::new(),
        }
    }

    #[test]
    fn contextual_tokens_and_legacy_wire_shape() {
        let result = convert_completion_logprobs(
            &payload(),
            &[1, 2],
            Arc::new(MockTokenizer::new()),
            true,
            "Hello world",
            0,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            serde_json::json!({
                "tokens": ["Hello", " world"],
                "token_logprobs": [-0.25, -1.5],
                "top_logprobs": [{"Hello": -0.25}, {" world": -1.5}],
                "text_offset": [0, 5]
            })
        );
    }

    #[test]
    fn empty_top_k_placeholders_and_ranked_candidates() {
        let mut proto = payload();
        proto.top_logprobs = vec![
            ProtoTopLogProbs {
                values: vec![],
                token_ids: vec![],
            },
            ProtoTopLogProbs {
                values: vec![-1.5, -2.0],
                token_ids: vec![2, 3],
            },
        ];
        let result = convert_completion_logprobs(
            &proto,
            &[1, 2],
            Arc::new(MockTokenizer::new()),
            true,
            "Hello world",
            0,
        )
        .unwrap();
        assert_eq!(result.top_logprobs[0].as_ref().unwrap()["Hello"], -0.25);
        let candidates = result.top_logprobs[1].as_ref().unwrap();
        assert_eq!(candidates[" world"], -1.5);
        assert_eq!(candidates["test"], -2.0);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn unicode_offsets_echo_and_stop_inside_token() {
        let result = build_logprobs(
            &payload(),
            vec!["é".into(), " answerSTOPhidden".into()],
            "é answer",
            2,
            |_| Err("unexpected candidate decoding".into()),
        )
        .unwrap();
        assert_eq!(result.tokens, ["é", " answer"]);
        assert_eq!(result.text_offset, [2, 3]);
        assert_eq!(result.top_logprobs[1].as_ref().unwrap()[" answer"], -1.5);
    }

    #[test]
    fn stop_hides_later_tokens_and_empty_output_has_no_scores() {
        for (text, expected) in [("Hello", vec!["Hello"]), ("", vec![])] {
            let result = convert_completion_logprobs(
                &payload(),
                &[1, 2],
                Arc::new(MockTokenizer::new()),
                true,
                text,
                0,
            )
            .unwrap();
            assert_eq!(result.tokens, expected);
            assert_eq!(result.token_logprobs.len(), result.tokens.len());
        }
    }

    #[test]
    fn incomplete_utf8_fragments_keep_their_position() {
        let result = build_logprobs(
            &payload(),
            vec![String::new(), "é".into()],
            "é",
            0,
            |_| Err("unexpected candidate decoding".into()),
        )
        .unwrap();
        assert_eq!(result.tokens, ["", "é"]);
        assert_eq!(result.text_offset, [0, 0]);
        assert_eq!(result.token_logprobs, [Some(-0.25), Some(-1.5)]);
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        let valid = payload();
        assert!(validate_payload(&valid, &[1, 3]).is_err());
        let mut bad = valid.clone();
        bad.token_logprobs.pop();
        assert!(validate_payload(&bad, &[1, 2]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            bad = valid.clone();
            bad.token_logprobs[0] = value;
            assert!(validate_payload(&bad, &[1, 2]).is_err());
        }
        bad = valid;
        bad.top_logprobs = vec![ProtoTopLogProbs {
            values: vec![],
            token_ids: vec![],
        }];
        assert!(validate_payload(&bad, &[1, 2]).is_err());
        bad.top_logprobs.push(ProtoTopLogProbs {
            values: vec![-2.0],
            token_ids: vec![],
        });
        assert!(validate_payload(&bad, &[1, 2]).is_err());
        bad.top_logprobs[1].token_ids.push(3);
        bad.top_logprobs[1].values[0] = f32::NAN;
        assert!(validate_payload(&bad, &[1, 2]).is_err());
    }

    #[test]
    fn text_mismatch_offset_overflow_and_candidate_decode_errors_propagate() {
        let tokens = vec!["A".into(), "é".into()];
        assert_eq!(
            build_logprobs(&payload(), tokens.clone(), "different", 0, |_| Err(
                "unexpected candidate decoding".into()
            ))
            .unwrap_err(),
            "Completion text does not match decoded logprob tokens"
        );
        assert_eq!(
            build_logprobs(&payload(), tokens.clone(), "Aé", usize::MAX, |_| Err(
                "unexpected candidate decoding".into()
            ))
            .unwrap_err(),
            "Completion text offset does not fit in u32"
        );
        let mut proto = payload();
        proto.top_logprobs = vec![ProtoTopLogProbs {
            values: vec![-2.0],
            token_ids: vec![3],
        }];
        assert_eq!(
            build_logprobs(&proto, tokens, "Aé", 0, |_| Err("decoder failed".into())).unwrap_err(),
            "decoder failed"
        );
    }
}
