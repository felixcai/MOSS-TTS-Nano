---
name: chat-note
description: Summarize current agent chat content into a markdown note file. Use when the user asks to整理/总结 agent chat notes, summarize from a specific point in the chat, or append conversation notes to a markdown file.
---

# Chat Note

## Purpose

整理当前 agent chat 中从用户指定位置开始的对话内容，并写入用户指定的 markdown 文件。

## When To Use

Use this skill when the user asks to:

- 总结当前 agent chat notes
- 整理从某次 chat 开始的内容
- 追加对话总结到某个 markdown 文件
- 按章节规则沉淀讨论、设计思路、开发思路或问题结论

## Workflow

1. Identify the requested start point.
   - If the user says “从上面这个对话开始”, start from the referenced message.
   - If the start point is ambiguous, ask a short clarification before editing.

2. Identify the target markdown file.
   - If the user provides a file path, use that file.
   - Read the file first if it already exists.
   - Preserve existing content unless the user explicitly asks to rewrite it.

3. Summarize only the relevant chat content.
   - Keep conclusions, decisions, definitions, open questions, and implementation rules.
   - Remove repeated back-and-forth, acknowledgements, and low-value phrasing.
   - Prefer concise engineering notes over transcript-style logs.

4. Organize chapters using the required rules:
   - If a topic or question is relatively independent, create a new H1 chapter.
   - If it supplements a previous topic or question, add it under that topic with a heading level one deeper than the parent heading.

5. Append or update the markdown file.
   - If the user says “追加”, append to the end.
   - If a matching chapter already exists and the new content is clearly a supplement, update that chapter instead of duplicating it.

## Section Rules

Use this structure by default:

```markdown
# 独立主题或问题

## 补充点或子问题

内容总结。

### 对上面 H2 小节的进一步补充

内容总结。
```

When appending a topic under an existing section, increment the heading level relative to the parent:

- Under `# H1`, use `## H2`.
- Under `## H2`, use `### H3`.
- Under `### H3`, use `#### H4`.

Do not create excessive heading levels. Prefer the shallowest level that accurately represents the parent-child relationship.

## Writing Style

- Use Chinese if the conversation is in Chinese.
- Keep notes actionable and specific.
- Use code fences for call chains, data structures, API shapes, and command examples.
- Use inline code for file paths, symbols, fields, and endpoint names.
- Avoid turning the note into a full transcript.

## Validation

After editing, check the markdown file for obvious formatting issues. If linter diagnostics are available, check the edited file.
