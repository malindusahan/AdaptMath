import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { describe, expect, it } from "vitest";
import { normalizeMathMarkdown } from "./mathMarkdown";

describe("normalizeMathMarkdown", () => {
  it("escapes repeated currency amounts so prose is not parsed as TeX", () => {
    const response =
      "For example, to find 15% of $240, you would convert 15% to 0.15 and multiply it by $240.";

    expect(normalizeMathMarkdown(response)).toBe(
      "For example, to find 15% of \\$240, you would convert 15% to 0.15 and multiply it by \\$240.",
    );

    const markup = renderToStaticMarkup(
      createElement(
        ReactMarkdown,
        { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] },
        normalizeMathMarkdown(response),
      ),
    );

    expect(markup).toContain(
      "For example, to find 15% of $240, you would convert 15% to 0.15 and multiply it by $240.",
    );
    expect(markup).not.toContain('class="katex"');
  });

  it("preserves genuine dollar-delimited inline mathematics", () => {
    const response = "Calculate $0.15 \\times 240 = 36$.";

    expect(normalizeMathMarkdown(response)).toBe(response);
  });

  it("continues to normalize parenthesized LaTeX delimiters", () => {
    expect(normalizeMathMarkdown("Calculate \\(0.15 \\times 240\\)."))
      .toBe("Calculate $0.15 \\times 240$.");
  });
});
