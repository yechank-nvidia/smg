//! Tool parser for a tokenizer's response template.

use async_trait::async_trait;
use openai_protocol::common::Tool;
use serde_json::Value;
use smg_response_template::{
    adapter::{self, Session},
    ResponseTemplate,
};

use crate::{
    errors::{ParserError, ParserResult},
    json_format,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Reads the `tool_calls` field of a response template with the transformers
/// parser of a [`Session`]. When the reasoning parser of the same output feeds
/// the session, the input is its content and the calls are the ones it
/// closed; otherwise this parser feeds the session and reasoning text stays in
/// the content. A call is reported whole, name and arguments, when its region
/// closes. Without an attached session the parser has no prompt, as
/// transformers with `prefix=""`.
pub struct TemplateToolParser {
    template: ResponseTemplate,
    session: Option<Session>,
    /// Calls reported so far.
    reported: usize,
    /// Calls closed at the end of the output.
    unstreamed: Vec<ToolCallItem>,
}

impl TemplateToolParser {
    pub fn new(template: ResponseTemplate) -> Self {
        Self {
            template,
            session: None,
            reported: 0,
            unstreamed: Vec::new(),
        }
    }

    fn new_session(&self, tools: &[Tool]) -> Session {
        let tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| serde_json::to_value(tool).ok())
            .collect();
        Session::new(&self.template, "", &tools, false)
    }

    fn items(&mut self, calls: Vec<adapter::ToolCall>) -> Vec<ToolCallItem> {
        calls
            .into_iter()
            .map(|call| {
                self.reported += 1;
                ToolCallItem {
                    tool_index: self.reported - 1,
                    name: Some(call.name),
                    parameters: arguments(call.arguments),
                }
            })
            .collect()
    }
}

/// `arguments` as a JSON string: kept when the template made it one.
fn arguments(value: Value) -> String {
    match value {
        Value::String(text) => text,
        value => json_format::to_string(&value),
    }
}

fn failed(error: smg_response_template::ParseError) -> ParserError {
    ParserError::ParsingFailed(format!("response template: {error}"))
}

#[async_trait]
impl ToolParser for TemplateToolParser {
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_with_tools(output, &[]).await
    }

    async fn parse_complete_with_tools(
        &self,
        output: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        let session = self
            .session
            .clone()
            .unwrap_or_else(|| self.new_session(tools));
        let (mut text, mut calls) = session.tools(Some(output)).map_err(failed)?;
        let (rest, more) = session.tools(None).map_err(failed)?;
        text.push_str(&rest);
        calls.extend(more);
        let calls = calls
            .into_iter()
            .map(|call| ToolCall {
                function: FunctionCall {
                    name: call.name,
                    arguments: arguments(call.arguments),
                },
            })
            .collect();
        Ok((text, calls))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        let session = match &self.session {
            Some(session) => session.clone(),
            None => self.session.insert(self.new_session(tools)).clone(),
        };
        let (normal_text, calls) = session.tools(Some(chunk)).unwrap_or_else(|error| {
            tracing::warn!("response template: {error}; the rest of the output is not parsed");
            (chunk.to_owned(), Vec::new())
        });
        Ok(StreamingParseResult {
            normal_text,
            calls: self.items(calls),
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        let session = self.new_session(&[]);
        [Some(text), None].into_iter().any(|text| {
            session
                .tools(text)
                .is_ok_and(|(_, calls)| !calls.is_empty())
        })
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        (!self.unstreamed.is_empty()).then(|| self.unstreamed.clone())
    }

    /// Ends the output: the rest of the content, and the calls it closed for
    /// `get_unstreamed_tool_args`.
    fn take_unstreamed_normal_text(&mut self) -> String {
        let Some(session) = self.session.clone() else {
            return String::new();
        };
        let (text, calls) = session.tools(None).unwrap_or_else(|error| {
            tracing::warn!("response template: {error}");
            (String::new(), Vec::new())
        });
        self.unstreamed = self.items(calls);
        text
    }

    fn reset(&mut self) {
        self.session = None;
        self.reported = 0;
        self.unstreamed.clear();
    }

    fn attach_response_session(&mut self, session: Session) {
        self.session = Some(session);
    }
}
