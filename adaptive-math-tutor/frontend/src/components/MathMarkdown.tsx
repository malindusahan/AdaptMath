import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

interface MathMarkdownProps {
  children: string;
}

function normalizeMathDelimiters(source: string): string {
  return source
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => {
      return `\n$$\n${expression.trim()}\n$$\n`;
    })
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => {
      return `$${expression.trim()}$`;
    });
}

export function MathMarkdown({ children }: MathMarkdownProps) {
  const normalized = useMemo(() => normalizeMathDelimiters(children), [children]);

  return (
    <div className="prose-adaptmath">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {normalized}
      </ReactMarkdown>
    </div>
  );
}
