//! Reasoning parser for a tokenizer's response template.

use smg_response_template::{adapter::Session, ResponseTemplate};

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

/// Reads the reasoning (`thinking` or `reasoning_content`) and `content`
/// fields of a response template with the transformers parser of a
/// [`Session`]. The tool parser of the same output takes the tool calls from
/// the session. Without an attached session the parser has no prompt, as
/// transformers with `prefix=""`.
pub struct TemplateReasoningParser {
    template: ResponseTemplate,
    session: Option<Session>,
    /// Bytes fed to the session.
    fed: usize,
}

impl TemplateReasoningParser {
    pub fn new(template: ResponseTemplate) -> Self {
        Self {
            template,
            session: None,
            fed: 0,
        }
    }

    fn session(&mut self) -> &Session {
        self.session
            .get_or_insert_with(|| Session::new(&self.template, "", &[], false))
    }

    fn check_size(&mut self, len: usize) -> Result<(), ParseError> {
        self.fed += len;
        if self.fed > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(self.fed));
        }
        Ok(())
    }
}

/// `ParseError` has no variant for output that fails to parse, so
/// `ConfigError` carries the template's error.
fn result(
    parsed: Result<(String, String), smg_response_template::ParseError>,
) -> Result<ParserResult, ParseError> {
    let (reasoning, content) =
        parsed.map_err(|e| ParseError::ConfigError(format!("response template: {e}")))?;
    Ok(ParserResult::new(content, reasoning))
}

impl ReasoningParser for TemplateReasoningParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.check_size(text.len())?;
        result(self.session().reasoning_complete(text))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.check_size(text.len())?;
        result(self.session().reasoning(Some(text)))
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        result(self.session().reasoning(None))
    }

    fn reset(&mut self) {
        self.session = None;
        self.fed = 0;
    }

    fn model_type(&self) -> &str {
        "response_template"
    }

    /// Templates delimit fields with special tokens as often as not;
    /// transformers parses output decoded with them.
    fn requires_special_tokens(&self) -> bool {
        true
    }

    fn is_in_reasoning(&self) -> bool {
        self.session.as_ref().is_some_and(Session::in_reasoning)
    }

    /// The session's prompt says where reasoning starts.
    fn mark_reasoning_started(&mut self) {}

    fn mark_think_start_stripped(&mut self) {}

    fn attach_response_session(&mut self, session: Session) {
        self.session = Some(session);
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use smg_response_template::load_response_template;

    use super::*;

    fn template() -> ResponseTemplate {
        load_response_template(&json!({
            "start_anchor": "<|assistant|>",
            "fields": {
                "thinking": {"open": "<think>", "close": "</think>"},
                "tool_calls": {
                    "open": "<call>", "close": "</call>", "repeats": true, "content": "json",
                    "transform": {"type": "function", "function": "{content}"}
                },
                "content": {"close": "<|end|>"}
            }
        }))
        .unwrap()
    }

    fn stream(parser: &mut TemplateReasoningParser, chunks: &[&str]) -> (String, String) {
        let (mut reasoning, mut content) = (String::new(), String::new());
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&result.reasoning_text);
            content.push_str(&result.normal_text);
        }
        let result = parser.flush().unwrap();
        reasoning.push_str(&result.reasoning_text);
        content.push_str(&result.normal_text);
        (reasoning, content)
    }

    #[test]
    fn the_prompt_decides_where_reasoning_starts() {
        let template = template();
        let mut parser = TemplateReasoningParser::new(template.clone());
        parser.attach_response_session(Session::new(&template, "<think>\n", &[], false));
        assert!(parser.is_in_reasoning());
        let chunks = ["plan", " a</thi", "nk>An", "swer<|end|>"];
        assert_eq!(
            stream(&mut parser, &chunks),
            ("plan a".to_owned(), "Answer".to_owned())
        );
        assert!(!parser.is_in_reasoning());

        // Without a session the output starts outside reasoning.
        let mut parser = TemplateReasoningParser::new(template);
        assert_eq!(
            stream(&mut parser, &chunks),
            (String::new(), "plan a</think>Answer".to_owned())
        );
    }

    #[test]
    fn a_complete_output_reads_the_parsed_message() {
        let mut parser = TemplateReasoningParser::new(template());
        let result = parser
            .detect_and_parse_reasoning(
                r#"<think> plan </think>first<call>{"name": "f", "arguments": {}}</call> last "#,
            )
            .unwrap();
        assert_eq!(result, ParserResult::new("last".into(), "plan".into()));
        parser.reset();
        let result = parser.detect_and_parse_reasoning("text").unwrap();
        assert_eq!(result, ParserResult::normal("text".into()));
    }

    #[test]
    fn after_an_error_the_output_passes_through() {
        let template = template();
        let mut parser = TemplateReasoningParser::new(template.clone());
        parser.attach_response_session(Session::new(&template, "", &[], false));
        let bad = parser.parse_reasoning_streaming_incremental("<call>{</call>");
        assert!(matches!(bad, Err(ParseError::ConfigError(_))));
        let result = parser
            .parse_reasoning_streaming_incremental("<think>x")
            .unwrap();
        assert_eq!(result, ParserResult::normal("<think>x".into()));
        assert_eq!(parser.flush().unwrap(), ParserResult::default());
    }
}
