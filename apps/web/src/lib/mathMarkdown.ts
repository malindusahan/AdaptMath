const CURRENCY_AMOUNT_PATTERN = /(?<!\\)\$(\d+(?:,\d{3})*(?:\.\d{1,2})?)/g;

function escapeCurrencyDollars(source: string): string {
  return source.replace(
    CURRENCY_AMOUNT_PATTERN,
    (match: string, _amount: string, offset: number, fullSource: string) => {
      const remainder = fullSource.slice(offset + match.length);

      // A closing dollar or a mathematical operator indicates genuine inline
      // TeX such as `$240$`, `$240 + 10$`, or `$15\%$`.
      if (/^\s*(?:\$|[+\-*/=<>^]|\\)/.test(remainder)) {
        return match;
      }

      // Dollar-prefixed numbers followed by prose, punctuation, or the end of
      // the response are currency. Escaping the dollar prevents remark-math
      // from pairing two prices into one accidental TeX expression.
      if (remainder.length === 0 || /^[\s,.;:!?)}\]]/.test(remainder)) {
        return `\\${match}`;
      }

      return match;
    },
  );
}

export function normalizeMathMarkdown(source: string): string {
  return escapeCurrencyDollars(source)
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, expression: string) => {
      return `\n$$\n${expression.trim()}\n$$\n`;
    })
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expression: string) => {
      return `$${expression.trim()}$`;
    });
}
