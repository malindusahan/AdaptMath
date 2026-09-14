import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { normalizeMathMarkdown } from "../lib/mathMarkdown";

interface MathMarkdownProps {
  children: string;
}

export function MathMarkdown({ children }: MathMarkdownProps) {
  const normalized = useMemo(() => normalizeMathMarkdown(children), [children]);

  return (
    <div className="prose-adaptmath">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
        {normalized}
      </ReactMarkdown>
    </div>
  );
}
